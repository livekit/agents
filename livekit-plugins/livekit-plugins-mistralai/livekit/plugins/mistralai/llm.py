from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ToolChoice,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given, shortuuid
from mistralai.client import Mistral
from mistralai.client.errors import HTTPValidationError, MistralError, SDKError
from mistralai.client.models import (
    CompletionArgs,
    ConversationEvents,
    FunctionCallEvent,
    MessageOutputEvent,
    ResponseDoneEvent,
    ResponseErrorEvent,
    ResponseStartedEvent,
    TextChunk,
    ToolExecutionDeltaEvent,
    ToolExecutionDoneEvent,
    ToolExecutionStartedEvent,
)
from mistralai.client.types import UNSET

from .log import logger
from .models import ChatModels
from .tools import MistralTool

DEFAULT_MODEL: ChatModels = "ministral-8b-latest"


class ApiMode(str, Enum):
    """Selects which Mistral API to use for chat requests.

    ``CONVERSATIONS`` uses the Conversations API (``/v1/conversations``),
    which supports both function tools and provider tools (web search,
    document library, code interpreter, connectors).

    ``CHAT_COMPLETIONS`` uses the Chat Completions API
    (``/v1/chat/completions``), which only supports function tools but
    has lower latency.
    """

    CONVERSATIONS = "conversations"
    CHAT_COMPLETIONS = "chat_completions"


@dataclass
class _LLMOptions:
    model: ChatModels | str
    api_mode: ApiMode
    max_completion_tokens: int | None
    temperature: float | None
    top_p: float | None
    presence_penalty: float | None
    frequency_penalty: float | None
    random_seed: int | None
    tool_choice: ToolChoice | None


@dataclass
class _PendingFunctionCall:
    """Accumulates streamed function call deltas."""

    id: str
    name: str
    tool_call_id: str
    arguments: str = ""


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        client: Mistral | None = None,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[ChatModels | str] = NOT_GIVEN,
        api_mode: NotGivenOr[ApiMode] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        frequency_penalty: NotGivenOr[float] = NOT_GIVEN,
        random_seed: NotGivenOr[int] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of MistralAI LLM.

        Args:
            client: Optional pre-configured MistralAI client instance.
            api_key: Your Mistral AI API key.  If not provided, will use
                the ``MISTRAL_API_KEY`` environment variable.
            model: The Mistral AI model to use, default is
                ``"ministral-8b-latest"``.
            api_mode: Which Mistral API to use.
                ``ApiMode.CONVERSATIONS`` (default) uses the Conversations
                API and supports both function tools and provider tools.
                ``ApiMode.CHAT_COMPLETIONS`` uses the Chat Completions API
                which only supports function tools but has lower latency.
            temperature: The temperature to use the LLM with.
            top_p: Nucleus sampling parameter.
            presence_penalty: Penalize new tokens based on their presence
                in the text so far.
            frequency_penalty: Penalize new tokens based on their
                frequency in the text so far.
            random_seed: Random seed for reproducibility.
            tool_choice: Default tool choice strategy
                (``"auto"``, ``"required"``, ``"none"``).
            max_completion_tokens: The max. number of tokens the LLM can
                output.
        """
        super().__init__()
        self._opts = _LLMOptions(
            model=model if is_given(model) else DEFAULT_MODEL,
            api_mode=api_mode if is_given(api_mode) else ApiMode.CONVERSATIONS,
            temperature=temperature if is_given(temperature) else None,
            top_p=top_p if is_given(top_p) else None,
            presence_penalty=presence_penalty if is_given(presence_penalty) else None,
            frequency_penalty=frequency_penalty if is_given(frequency_penalty) else None,
            random_seed=random_seed if is_given(random_seed) else None,
            tool_choice=tool_choice if is_given(tool_choice) else None,
            max_completion_tokens=max_completion_tokens
            if is_given(max_completion_tokens)
            else None,
        )

        mistral_api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not client and not mistral_api_key:
            raise ValueError("Mistral AI API key is required. Set MISTRAL_API_KEY or pass api_key")
        self._client = client or Mistral(api_key=mistral_api_key)

    async def _prewarm_impl(self) -> None:
        await self._client.models.list_async()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MistralAI"

    def update_options(
        self,
        *,
        model: NotGivenOr[ChatModels | str] = NOT_GIVEN,
        api_mode: NotGivenOr[ApiMode] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        frequency_penalty: NotGivenOr[float] = NOT_GIVEN,
        random_seed: NotGivenOr[int] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(api_mode):
            self._opts.api_mode = api_mode
        if is_given(max_completion_tokens):
            self._opts.max_completion_tokens = max_completion_tokens
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(top_p):
            self._opts.top_p = top_p
        if is_given(presence_penalty):
            self._opts.presence_penalty = presence_penalty
        if is_given(frequency_penalty):
            self._opts.frequency_penalty = frequency_penalty
        if is_given(random_seed):
            self._opts.random_seed = random_seed
        if is_given(tool_choice):
            self._opts.tool_choice = tool_choice

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra: dict[str, Any] = {}
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        completion_args: dict[str, Any] = {}
        if self._opts.max_completion_tokens is not None:
            completion_args["max_tokens"] = self._opts.max_completion_tokens
        if self._opts.temperature is not None:
            completion_args["temperature"] = self._opts.temperature
        if self._opts.top_p is not None:
            completion_args["top_p"] = self._opts.top_p
        if self._opts.presence_penalty is not None:
            completion_args["presence_penalty"] = self._opts.presence_penalty
        if self._opts.frequency_penalty is not None:
            completion_args["frequency_penalty"] = self._opts.frequency_penalty
        if self._opts.random_seed is not None:
            completion_args["random_seed"] = self._opts.random_seed

        resolved_tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        if resolved_tool_choice is not None:
            has_provider_tools = any(isinstance(t, MistralTool) for t in (tools or []))
            if isinstance(resolved_tool_choice, dict) or resolved_tool_choice == "required":
                completion_args["tool_choice"] = "auto" if has_provider_tools else "required"
            elif resolved_tool_choice in ("auto", "none"):
                completion_args["tool_choice"] = resolved_tool_choice
        if completion_args:
            extra["completion_args"] = CompletionArgs(**completion_args)

        return LLMStream(
            self,
            model=self._opts.model,
            api_mode=self._opts.api_mode,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
            tool_choice=resolved_tool_choice,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_v: LLM,
        *,
        model: str | ChatModels,
        api_mode: ApiMode,
        client: Mistral,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        extra_kwargs: dict[str, Any],
        tool_choice: ToolChoice | None = None,
    ) -> None:
        super().__init__(llm_v, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._api_mode = api_mode
        self._client = client
        self._extra_kwargs = extra_kwargs
        self._tool_choice = tool_choice
        self._tool_ctx = llm.ToolContext(tools)
        self._emitted_tool_calls: set[str] = set()
        self._provider_tool_args: dict[str, str] = {}

    async def _run(self) -> None:
        if self._api_mode == ApiMode.CHAT_COMPLETIONS:
            await self._run_chat_completions()
        else:
            await self._run_conversations()

    async def _run_conversations(self) -> None:
        self._emitted_tool_calls = set()
        self._provider_tool_args = {}
        retryable = True

        try:
            entries, extra_data = self._chat_ctx.to_provider_format(format="mistralai")
            tools_list = self._tool_ctx.parse_function_tools("openai", strict=True)
            for tool in self._tool_ctx.provider_tools:
                if isinstance(tool, MistralTool):
                    tools_list.append(tool.to_dict())

            start_kwargs: dict[str, Any] = {}
            if tools_list:
                start_kwargs["tools"] = tools_list

            async_response = await self._client.beta.conversations.start_stream_async(
                inputs=entries,
                model=self._model,
                instructions=extra_data.instructions,
                timeout_ms=int(self._conn_options.timeout * 1000),
                **start_kwargs,
                **self._extra_kwargs,
            )

            pending_fnc_calls: dict[str, _PendingFunctionCall] = {}

            async for ev in async_response:
                chunks = self._parse_event(ev, pending_fnc_calls)
                for chat_chunk in chunks:
                    retryable = False
                    self._event_ch.send_nowait(chat_chunk)

            for chat_chunk in self._flush_pending_fnc_calls(pending_fnc_calls):
                self._event_ch.send_nowait(chat_chunk)

        except APITimeoutError:
            raise APITimeoutError(retryable=retryable) from None
        except (asyncio.TimeoutError, httpx.TimeoutException) as e:
            raise APITimeoutError(retryable=retryable) from e
        except APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=retryable,
            ) from None
        except (SDKError, HTTPValidationError) as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.headers.get("x-request-id"),
                body=e.body,
                retryable=retryable,
            ) from e
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    async def _run_chat_completions(self) -> None:
        if any(isinstance(t, MistralTool) for t in self._tool_ctx.provider_tools):
            raise ValueError(
                "Provider tools (WebSearch, DocumentLibrary, CodeInterpreter, "
                "Connector) are not supported with api_mode='chat_completions'. "
                "Use api_mode='conversations' or remove provider tools."
            )

        retryable = True

        try:
            messages, _ = self._chat_ctx.to_provider_format(format="openai")
            tools_list = self._tool_ctx.parse_function_tools("openai", strict=True)

            extra_kwargs = dict(self._extra_kwargs)
            completion_args = extra_kwargs.pop("completion_args", None)

            call_kwargs: dict[str, Any] = {}
            if completion_args is not None:
                for field in (
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "presence_penalty",
                    "frequency_penalty",
                    "random_seed",
                ):
                    val = getattr(completion_args, field, None)
                    if val is not None:
                        call_kwargs[field] = val

            if self._tool_choice is not None:
                call_kwargs["tool_choice"] = self._tool_choice

            if tools_list:
                call_kwargs["tools"] = tools_list

            call_kwargs.update(extra_kwargs)

            async_response = await self._client.chat.stream_async(
                model=self._model,
                messages=messages,
                timeout_ms=int(self._conn_options.timeout * 1000),
                **call_kwargs,
            )

            pending_fnc_calls: dict[int, _PendingFunctionCall] = {}

            async for ev in async_response:
                chunk = ev.data
                for choice in chunk.choices:
                    chat_chunks = _parse_completion_delta(
                        chunk_id=chunk.id,
                        delta=choice.delta,
                        finish_reason=choice.finish_reason,
                        pending_fnc_calls=pending_fnc_calls,
                    )
                    for cc in chat_chunks:
                        retryable = False
                        self._event_ch.send_nowait(cc)

                if chunk.usage is not None:
                    self._event_ch.send_nowait(
                        ChatChunk(
                            id=chunk.id,
                            usage=llm.CompletionUsage(
                                completion_tokens=chunk.usage.completion_tokens or 0,
                                prompt_tokens=chunk.usage.prompt_tokens or 0,
                                total_tokens=chunk.usage.total_tokens or 0,
                            ),
                        )
                    )

            for cc in _flush_pending_by_index(pending_fnc_calls):
                self._event_ch.send_nowait(cc)

        except APITimeoutError:
            raise APITimeoutError(retryable=retryable) from None
        except (asyncio.TimeoutError, httpx.TimeoutException) as e:
            raise APITimeoutError(retryable=retryable) from e
        except APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=retryable,
            ) from None
        except MistralError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=None,
                body=e.body,
                retryable=retryable and (e.status_code in (408, 429) or e.status_code >= 500),
            ) from e
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _flush_pending_fnc_calls(self, pending: dict[str, _PendingFunctionCall]) -> list[ChatChunk]:
        """Emit completed FunctionToolCalls from the pending buffer."""
        chunks: list[ChatChunk] = []
        for fnc in pending.values():
            delta = llm.ChoiceDelta(role="assistant")
            delta.tool_calls.append(
                llm.FunctionToolCall(
                    name=fnc.name,
                    arguments=fnc.arguments,
                    call_id=fnc.tool_call_id,
                )
            )
            chunks.append(ChatChunk(id=fnc.id, delta=delta))
            self._emitted_tool_calls.add(fnc.tool_call_id)
        pending.clear()
        return chunks

    def _parse_event(
        self,
        ev: ConversationEvents,
        pending_fnc_calls: dict[str, _PendingFunctionCall],
    ) -> list[ChatChunk]:
        data = ev.data
        chunks: list[ChatChunk] = []

        if isinstance(data, ResponseStartedEvent):
            return chunks

        if isinstance(data, FunctionCallEvent):
            # accumulate arguments across delta events
            call_id = data.tool_call_id or shortuuid("tool_call_")
            if call_id not in pending_fnc_calls:
                pending_fnc_calls[call_id] = _PendingFunctionCall(
                    id=data.id,
                    name=data.name,
                    tool_call_id=call_id,
                )
            pending_fnc_calls[call_id].arguments += data.arguments
            return chunks

        # any non-FunctionCallEvent flushes pending function calls
        chunks.extend(self._flush_pending_fnc_calls(pending_fnc_calls))

        if isinstance(data, MessageOutputEvent):
            content = data.content
            if isinstance(content, str):
                text = content
            elif isinstance(content, TextChunk):
                text = content.text
            else:
                return chunks

            if text:
                chunks.append(
                    ChatChunk(
                        id=data.id,
                        delta=llm.ChoiceDelta(content=text, role="assistant"),
                    )
                )
            return chunks

        if isinstance(data, ResponseDoneEvent):
            usage = data.usage
            chunks.append(
                ChatChunk(
                    id=shortuuid("done_"),
                    usage=llm.CompletionUsage(
                        completion_tokens=usage.completion_tokens or 0,
                        prompt_tokens=usage.prompt_tokens or 0,
                        total_tokens=usage.total_tokens or 0,
                    ),
                )
            )
            return chunks

        if isinstance(data, ResponseErrorEvent):
            raise APIStatusError(
                data.message,
                status_code=data.code,
                request_id=None,
                body=None,
                retryable=False,
            )

        if isinstance(data, ToolExecutionStartedEvent):
            self._provider_tool_args[data.id] = data.arguments

        elif isinstance(data, ToolExecutionDeltaEvent):
            if data.id not in self._provider_tool_args:
                self._provider_tool_args[data.id] = ""
            self._provider_tool_args[data.id] += data.arguments

        elif isinstance(data, ToolExecutionDoneEvent):
            args = self._provider_tool_args.pop(data.id, "")
            logger.debug(
                "executed provider tool",
                extra={"function": data.name, "lk.pii.arguments": args, "lk.pii.info": data.info},
            )

        return chunks


def _parse_completion_delta(
    *,
    chunk_id: str,
    delta: Any,
    finish_reason: str | None,
    pending_fnc_calls: dict[int, _PendingFunctionCall],
) -> list[ChatChunk]:
    chunks: list[ChatChunk] = []

    content = delta.content
    if content is not UNSET and content is not None:
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(c.text for c in content if isinstance(c, TextChunk))
        else:
            text = ""
        if text:
            chunks.append(
                ChatChunk(
                    id=chunk_id,
                    delta=llm.ChoiceDelta(content=text, role="assistant"),
                )
            )

    tool_calls = delta.tool_calls
    if tool_calls is not UNSET and tool_calls is not None:
        for tc in tool_calls:
            if tc.index is not None:
                call_index = tc.index
            else:
                call_index = max(pending_fnc_calls.keys(), default=-1) + 1
            call_id = tc.id if tc.id not in (None, "null") else ""
            name = tc.function.name if tc.function else ""
            args = tc.function.arguments if tc.function else ""
            if isinstance(args, dict):
                args = json.dumps(args)

            if call_index not in pending_fnc_calls:
                pending_fnc_calls[call_index] = _PendingFunctionCall(
                    id=chunk_id,
                    name=name,
                    tool_call_id=call_id,
                    arguments=args,
                )
            else:
                if name:
                    pending_fnc_calls[call_index].name = name
                pending_fnc_calls[call_index].arguments += args

    if finish_reason == "tool_calls":
        chunks.extend(_flush_pending_by_index(pending_fnc_calls))

    return chunks


def _flush_pending_by_index(
    pending: dict[int, _PendingFunctionCall],
) -> list[ChatChunk]:
    chunks: list[ChatChunk] = []
    for fnc in pending.values():
        delta = llm.ChoiceDelta(role="assistant")
        delta.tool_calls.append(
            llm.FunctionToolCall(
                name=fnc.name,
                arguments=fnc.arguments,
                call_id=fnc.tool_call_id,
            )
        )
        chunks.append(ChatChunk(id=fnc.id, delta=delta))
    pending.clear()
    return chunks
