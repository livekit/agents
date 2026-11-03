# Copyright 2026 LiveKit, Inc.
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

from typing import Any

import httpx2 as httpx
import openai

from livekit.agents.llm import ChatContext, Tool, ToolChoice
from livekit.agents.llm._provider_format.utils import convert_mid_conversation_instructions
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM
from livekit.plugins.openai.llm import LLMStream

from ._common import client_headers, openai_base_url, resolve_api_key
from .models import SUPPORTED_SAMPLING, ChatModels

# The gateway can hold a streamed reply until the first token, and allows qwen/qwen3.8-27b up to
# 60 s for it (https://docs.vlm.run/gateway/error-codes). The openai plugin's 5 s read timeout
# failed about 1 in 10 turns on that model, so wait past the gateway's own deadline and let its
# 504 arrive first.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=15.0, read=70.0, write=5.0, pool=5.0)


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "qwen/qwen3.8-27b",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        extra_body: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of a VLM Run gateway LLM.

        The VLM Run gateway serves open-weight and hosted vision-language models through an
        OpenAI-compatible Chat Completions API. Images in the chat context (``ImageContent``,
        including an ``rtc.VideoFrame``) are sent to the model. An ``AgentSession`` forwards
        live video only to realtime models, so the agent must add camera frames to the chat
        context itself; see the plugin README.

        ``api_key`` defaults to the ``VLMRUN_API_KEY`` environment variable and is required.
        ``base_url`` defaults to ``VLMRUN_GATEWAY_BASE_URL``, then
        ``VLMRUN_GATEWAY_URL``, then ``https://gateway.vlm.run``.

        Tool calling is supported by ``qwen/qwen3.8-27b`` and the hosted models (Gemini, Kimi,
        MiniMax, Muse), not by ``qwen/qwen3.5-0.8b`` or the Gemma models.

        Only parameters you set are sent. ``temperature`` and ``top_p`` are checked against what
        the model accepts, since the gateway rejects unsupported sampling fields.

        Args:
            model: Gateway model id. Defaults to ``qwen/qwen3.8-27b``.
            api_key: VLM Run API key. Defaults to ``VLMRUN_API_KEY``.
            base_url: Gateway root URL, without the ``/openai`` suffix.
            client: A preconfigured ``openai.AsyncClient``; overrides key, URL and timeout.
            temperature: Sampling temperature.
            top_p: Nucleus sampling.
            parallel_tool_calls: Allow parallel tool calls.
            tool_choice: Tool choice strategy.
            max_completion_tokens: Maximum tokens per reply.
            extra_body: Gateway-specific fields, such as ``method`` or ``method_params``.
            timeout: HTTP timeout. The default read timeout is 70 s, above the gateway's
                first-token deadline.
            max_retries: Retries made by the openai client.
        """
        _check_sampling(model, temperature=temperature, top_p=top_p)

        key = resolve_api_key(api_key if is_given(api_key) else None)
        created_client = False
        if client is None:
            if not key:
                raise ValueError(
                    "VLMRUN_API_KEY is required, either as argument or set VLMRUN_API_KEY environment variable"  # noqa: E501
                )
            client = openai.AsyncClient(
                api_key=key,
                base_url=openai_base_url(base_url if is_given(base_url) else None),
                default_headers=client_headers(None),
                max_retries=max_retries if is_given(max_retries) else 0,
                http_client=httpx.AsyncClient(
                    timeout=timeout if timeout else _DEFAULT_TIMEOUT,
                    follow_redirects=True,
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=50,
                        keepalive_expiry=120,
                    ),
                ),
            )
            created_client = True

        super().__init__(
            model=model,
            client=client,
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_completion_tokens=max_completion_tokens,
            extra_body=extra_body,
            _strict_tool_schema=False,
        )

        if created_client:
            self._owns_client = True

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return super().chat(
            chat_ctx=_gateway_chat_ctx(chat_ctx),
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_kwargs=extra_kwargs,
        )


def _check_sampling(
    model: str, *, temperature: NotGivenOr[float], top_p: NotGivenOr[float]
) -> None:
    supported = SUPPORTED_SAMPLING.get(model)
    if supported is None:
        return
    for name, value in (("temperature", temperature), ("top_p", top_p)):
        if is_given(value) and name not in supported:
            raise ValueError(f"VLM Run model {model!r} does not accept {name!r}")


def _gateway_chat_ctx(chat_ctx: ChatContext) -> ChatContext:
    """Shape a conversation the way the gateway's chat models accept it.

    The gateway rejects a system message anywhere but first ("System message must be at
    the beginning") and a request with no user message ("No user query found in
    messages"). generate_reply(instructions=...) produces both: a later system message,
    and on an opening greeting no user turn at all. Later instructions become inline user
    turns where they fall, as the anthropic, aws, google and mistralai formats do, and
    "(empty)" is added as the user turn when there is still none.
    """
    shaped = convert_mid_conversation_instructions(chat_ctx)
    if not any(item.type == "message" and item.role == "user" for item in shaped.items):
        shaped.add_message(role="user", content="(empty)")
    return shaped
