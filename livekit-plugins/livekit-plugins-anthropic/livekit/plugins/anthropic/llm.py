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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx

import anthropic
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import Tool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import (
    ChatModels,
    _model_supports_prefill,
    _model_supports_sampling_params,
    _model_thinking_support,
)
from .utils import CACHE_CONTROL_EPHEMERAL

# Rejected by Claude 4.7 and later. `top_p` has no constructor argument: it can only
# reach the request through `extra_kwargs`.
_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")

# `max_tokens` is a single ceiling over thinking *and* the reply. 1024 is plenty for a
# spoken turn, but a model that always thinks would spend most of it reasoning and
# return a truncated (or empty) answer, so those get more headroom.
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_MAX_TOKENS_THINKING = 8192


@dataclass
class _LLMOptions:
    model: str | ChatModels
    user: NotGivenOr[str]
    temperature: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]
    caching: NotGivenOr[Literal["ephemeral"]]
    top_k: NotGivenOr[int]
    max_tokens: NotGivenOr[int]
    strict_tool_schema: bool
    """If set to "ephemeral", the system prompt, tools, and chat history will be cached."""


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "claude-sonnet-4-6",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        user: NotGivenOr[str] = NOT_GIVEN,
        client: anthropic.AsyncClient | None = None,
        top_k: NotGivenOr[int] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        caching: NotGivenOr[Literal["ephemeral"]] = NOT_GIVEN,
        timeout: NotGivenOr[httpx.Timeout] = NOT_GIVEN,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
        _strict_tool_schema: bool = True,
    ) -> None:
        """
        Create a new instance of Anthropic LLM.

        ``api_key`` must be set to your Anthropic API key, either using the argument or by setting
        the ``ANTHROPIC_API_KEY`` environmental variable.

        model (str | ChatModels): The model to use. Defaults to "claude-sonnet-4-6".
        api_key (str, optional): The Anthropic API key. Defaults to the ANTHROPIC_API_KEY environment variable.
        base_url (str, optional): The base URL for the Anthropic API. Defaults to None.
        user (str, optional): The user for the Anthropic API. Defaults to None.
        client (anthropic.AsyncClient | None): The Anthropic client to use. Defaults to None.
        max_retries (int, optional): Vendor client retries. Defaults to 0 because the framework
            owns retries via ``conn_options``.
        timeout (httpx.Timeout | None): HTTP timeout configuration for the underlying httpx client.
            Defaults to ``httpx.Timeout(5.0, read=30.0)``, which keeps a tight connect timeout
            while allowing up to 30 s between streamed chunks — long enough for Claude's
            adaptive-thinking phases without masking genuine network stalls.
            Pass a custom ``httpx.Timeout`` to override (e.g. ``httpx.Timeout(5.0, read=60.0)``
            for very large contexts or extended thinking budgets).
        temperature (float, optional): The temperature for the Anthropic API. Defaults to None.
            Claude 4.7 and later reject sampling parameters; on those models ``temperature``
            and ``top_k`` are dropped from the request (with a one-time warning) instead of
            failing it. Steer those models with the system prompt instead.
        parallel_tool_calls (bool, optional): Whether to parallelize tool calls. Defaults to None.
        tool_choice (ToolChoice, optional): The tool choice for the Anthropic API. Defaults to "auto".
        caching (Literal["ephemeral"], optional): If set to "ephemeral", caching will be enabled for the system prompt, tools, and chat history.

        Thinking is disabled by default on every model that accepts the parameter: a voice
        agent pays for it in latency and in ``max_tokens``, which caps thinking and the
        reply together. Pass ``extra_kwargs={"thinking": {"type": "adaptive"}}`` to
        ``chat()`` to opt back in — but note that thinking blocks are not kept in the chat
        context, so a turn carrying a tool call is replayed without them, which Anthropic
        can reject.
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
            strict_tool_schema=_strict_tool_schema,
        )
        self._sampling_params_warned = False
        self._thinking_tools_warned = False
        anthropic_api_key = api_key if is_given(api_key) else os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError(
                "Anthropic API key is required, either as argument or set"
                " ANTHROPIC_API_KEY environment variable"
            )

        self._client = client or anthropic.AsyncClient(
            api_key=anthropic_api_key,
            base_url=base_url if is_given(base_url) else None,
            max_retries=max_retries if is_given(max_retries) else 0,
            http_client=httpx.AsyncClient(
                timeout=timeout or httpx.Timeout(5.0, read=30.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                    keepalive_expiry=120,
                ),
            ),
        )

    async def _prewarm_impl(self) -> None:
        await self._client.models.list(limit=1)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return self._client._base_url.netloc.decode("utf-8")

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra = {}

        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.user):
            extra["user"] = self._opts.user

        self._apply_sampling_params(extra)

        # Thinking costs latency and eats into max_tokens, so it is off wherever it can
        # be turned off. `extra_kwargs` wins: it is the escape hatch for callers that do
        # want the model to reason before answering.
        thinking_support = _model_thinking_support(self._opts.model)
        thinking: Any = extra.get("thinking")
        if thinking is None and thinking_support == "configurable":
            thinking = extra["thinking"] = {"type": "disabled"}

        thinking_enabled = thinking_support == "always_on" or (
            isinstance(thinking, dict) and thinking.get("type") != "disabled"
        )

        if thinking_enabled and tools and not self._thinking_tools_warned:
            self._thinking_tools_warned = True
            logger.warning(
                "%s will reason before answering, but thinking blocks are not kept in the "
                "chat context: the next turn replays the assistant turn without them, "
                "which Anthropic can reject when that turn contains a tool call",
                self._opts.model,
            )

        if is_given(self._opts.max_tokens):
            extra["max_tokens"] = self._opts.max_tokens
        elif "max_tokens" not in extra:
            extra["max_tokens"] = (
                _DEFAULT_MAX_TOKENS_THINKING if thinking_enabled else _DEFAULT_MAX_TOKENS
            )

        beta_flag: str | None = None
        if tools:
            from .tools import AnthropicTool

            tool_ctx = llm.ToolContext(tools)
            tool_schemas = tool_ctx.parse_function_tools(
                "anthropic", strict=self._opts.strict_tool_schema
            )

            for tool in tool_ctx.provider_tools:
                if isinstance(tool, AnthropicTool):
                    tool_schemas.append(tool.to_dict())
                    if tool.beta_flag:
                        beta_flag = tool.beta_flag

            extra["tools"] = tool_schemas

            tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
            if is_given(tool_choice):
                anthropic_tool_choice: dict[str, Any] | None = {"type": "auto"}
                if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                    anthropic_tool_choice = {
                        "type": "tool",
                        "name": tool_choice["function"]["name"],
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

        # Claude 4.6+ does not support prefilling (trailing assistant messages).
        inject_trailing = not _model_supports_prefill(self._opts.model)
        anthropic_ctx, extra_data = chat_ctx.to_provider_format(
            format="anthropic", inject_trailing_user_message=inject_trailing
        )
        messages = cast(list[anthropic.types.MessageParam], anthropic_ctx)
        if extra_data.system_messages:
            extra["system"] = [
                anthropic.types.TextBlockParam(text=content, type="text")
                for content in extra_data.system_messages
            ]

        # add cache control
        if self._opts.caching == "ephemeral":
            if extra.get("system"):
                extra["system"][-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL

            if extra.get("tools"):
                extra["tools"][-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL

            seen_assistant = False
            for msg in reversed(messages):
                if (
                    msg["role"] == "assistant"
                    and (content := msg["content"])
                    and not seen_assistant
                ):
                    content[-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL  # type: ignore
                    seen_assistant = True

                elif msg["role"] == "user" and (content := msg["content"]) and seen_assistant:
                    content[-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL  # type: ignore
                    break

        async def create_anthropic_stream() -> anthropic.AsyncStream[
            anthropic.types.RawMessageStreamEvent
        ]:
            if beta_flag:
                stream = await self._client.beta.messages.create(
                    betas=[beta_flag],
                    messages=messages,  # type: ignore[arg-type]
                    model=self._opts.model,
                    stream=True,
                    timeout=conn_options.timeout,
                    **extra,
                )
            else:
                stream = await self._client.messages.create(
                    messages=messages,
                    model=self._opts.model,
                    stream=True,
                    timeout=conn_options.timeout,
                    **extra,
                )
            return cast(anthropic.AsyncStream[anthropic.types.RawMessageStreamEvent], stream)

        return LLMStream(
            self,
            create_anthropic_stream=create_anthropic_stream,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            thinking_expected=thinking_enabled,
        )

    def _apply_sampling_params(self, extra: dict[str, Any]) -> None:
        """Add `temperature`/`top_k` to the request, unless the model rejects them.

        Claude 4.7 and later answer a request carrying sampling parameters with a 400, so
        they are dropped instead — including the ones that came in through
        ``extra_kwargs``, which are already in ``extra`` by the time this runs. The
        warning is logged once per instance to stay visible without flooding a
        long-running session.
        """
        if _model_supports_sampling_params(self._opts.model):
            if is_given(self._opts.temperature):
                extra["temperature"] = self._opts.temperature

            if is_given(self._opts.top_k):
                extra["top_k"] = self._opts.top_k

            return

        from_extra = {name for name in _SAMPLING_PARAMS if name in extra}
        for name in from_extra:
            del extra[name]

        from_opts = {
            name
            for name, value in (
                ("temperature", self._opts.temperature),
                ("top_k", self._opts.top_k),
            )
            if is_given(value)
        }
        dropped = [name for name in _SAMPLING_PARAMS if name in from_extra | from_opts]
        if dropped and not self._sampling_params_warned:
            self._sampling_params_warned = True
            logger.warning(
                "%s rejects sampling parameters, dropping %s from the request; "
                "steer the model with the system prompt instead",
                self._opts.model,
                ", ".join(dropped),
            )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        create_anthropic_stream: Callable[
            [], Awaitable[anthropic.AsyncStream[anthropic.types.RawMessageStreamEvent]]
        ],
        chat_ctx: llm.ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        thinking_expected: bool = False,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._create_anthropic_stream = create_anthropic_stream
        self._thinking_expected = thinking_expected

        # current function call that we're waiting for full completion (args are streamed)
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None

        self._request_id: str = ""
        self._ignoring_cot = False  # ignore chain of thought
        self._thinking_blocks = 0
        self._thinking_chars = 0
        self._input_tokens = 0
        self._cache_creation_tokens = 0
        self._cache_read_tokens = 0
        self._output_tokens = 0

    def _reset_stream_state(self) -> None:
        """Clear the state a single attempt builds up.

        The base class re-enters `_run` on the same instance, and it only retries when the
        failed attempt emitted no chunk — a stream that died mid-thinking, mid
        chain-of-thought or mid tool-call would otherwise carry that state into the next
        attempt, including the `_ignoring_cot` latch that swallows text. `_input_tokens`
        and `_output_tokens` are reassigned unconditionally at `message_start`, but the
        cache counters are only assigned there when the new attempt reports a non-zero
        value, so they are cleared here.
        """
        self._tool_call_id = None
        self._fnc_name = None
        self._fnc_raw_arguments = None
        self._ignoring_cot = False
        self._thinking_blocks = 0
        self._thinking_chars = 0
        self._cache_creation_tokens = 0
        self._cache_read_tokens = 0

    async def _run(self) -> None:
        retryable = True
        self._reset_stream_state()
        try:
            async with await self._create_anthropic_stream() as stream:
                async for event in stream:
                    chat_chunk = self._parse_event(event)
                    if chat_chunk is not None:
                        self._event_ch.send_nowait(chat_chunk)
                        retryable = False

                if self._thinking_blocks:
                    logger.debug(
                        "anthropic: %s returned %d thinking block(s), %d characters",
                        self._llm.model,
                        self._thinking_blocks,
                        self._thinking_chars,
                    )

                # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
                prompt_token = (
                    self._input_tokens + self._cache_creation_tokens + self._cache_read_tokens
                )
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=self._request_id,
                        usage=llm.CompletionUsage(
                            completion_tokens=self._output_tokens,
                            prompt_tokens=prompt_token,
                            total_tokens=prompt_token + self._output_tokens,
                            prompt_cached_tokens=self._cache_read_tokens,
                            cache_creation_tokens=self._cache_creation_tokens,
                            cache_read_tokens=self._cache_read_tokens,
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
            elif event.content_block.type in ("thinking", "redacted_thinking"):
                self._note_thinking_block()
        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "thinking_delta":
                # Reasoning is not part of the answer: it must not reach the caller (a
                # voice agent would speak it), but it is billed and it is counted below.
                self._thinking_chars += len(delta.thinking)
                return None
            elif delta.type == "signature_delta":
                return None
            elif delta.type == "text_delta":
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

    def _note_thinking_block(self) -> None:
        """Record a thinking block, warning once when one was not expected."""
        self._thinking_blocks += 1
        if self._thinking_blocks == 1 and not self._thinking_expected:
            logger.warning(
                "anthropic: %s returned a thinking block although thinking was not "
                "requested; the reasoning is dropped but still counts against max_tokens",
                self._llm.model,
            )
