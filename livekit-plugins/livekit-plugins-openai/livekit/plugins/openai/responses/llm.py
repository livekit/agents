from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx2 as httpx
from yarl import URL

import openai
from livekit.agents import llm
from livekit.agents.inference.llm import (
    PromptCacheOptions,
    supports_prompt_cache_breakpoints,
)
from livekit.agents.llm import ToolChoice
from livekit.agents.llm._responses_engine import (
    _OPENAI_HOST,
    _WS_HEARTBEAT,  # noqa: F401
    OPENAI_RESPONSES_WS_URL,  # noqa: F401
    ResponsesLLMStream as LLMStream,
    ResponsesWebsocket as _ResponsesWebsocket,
)
from livekit.agents.llm.chat_context import ChatContext, ChatItem
from livekit.agents.llm.tool_context import (
    Tool,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types import Reasoning
from openai.types.responses import (
    response_create_params,
)
from openai.types.shared_params import ResponsesModel

from ..log import logger
from ..models import _supports_reasoning_effort
from ..tools import OpenAITool

ServiceTier = Literal["auto", "default", "flex", "scale", "priority", "ultrafast"]
Verbosity = Literal["low", "medium", "high"]


@dataclass
class _LLMOptions:
    model: str | ResponsesModel
    user: NotGivenOr[str]
    temperature: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]]
    store: NotGivenOr[bool]
    reasoning: NotGivenOr[Reasoning]
    metadata: NotGivenOr[dict[str, str]]
    service_tier: NotGivenOr[ServiceTier]
    verbosity: NotGivenOr[Verbosity]
    max_output_tokens: NotGivenOr[int]
    use_websocket: bool
    prompt_cache_options: NotGivenOr[PromptCacheOptions]
    prompt_cache_breakpoints: bool | Literal["auto"]


class LLM(llm.LLM):
    # the plugin's ProviderTool subclass; subclasses (e.g. xAI) override this so server-side
    # provider tools are recognized when serializing the request. See to_responses_fnc_ctx.
    _provider_tool_type: type[llm.ProviderTool] = OpenAITool

    def __init__(
        self,
        *,
        model: str | ResponsesModel = "gpt-4.1",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        use_websocket: bool = True,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        reasoning: NotGivenOr[Reasoning] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]] = NOT_GIVEN,
        store: NotGivenOr[bool] = NOT_GIVEN,
        metadata: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        service_tier: NotGivenOr[ServiceTier] = NOT_GIVEN,
        verbosity: NotGivenOr[Verbosity] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        prompt_cache_options: NotGivenOr[PromptCacheOptions] = NOT_GIVEN,
        prompt_cache_breakpoints: bool | Literal["auto"] = "auto",
    ) -> None:
        """
        Create a new instance of OpenAI Responses LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.

        ``prompt_cache_breakpoints`` sends each :class:`livekit.agents.llm.CacheBreakpoint` in the
        chat context as an OpenAI ``prompt_cache_breakpoint``. ``"auto"`` sends them only against
        api.openai.com and only for GPT-5.6 and later. ``True`` sends them anyway, for a custom host
        that accepts the field; ``False`` never sends them. Models before GPT-5.6 reject the field
        with HTTP 400, so ``True`` cannot add support to them.

        ``prompt_cache_options`` is forwarded as given on every request; it is not gated like the
        breakpoints, and models before GPT-5.6 reject it the same way.
        """
        super().__init__()

        if not is_given(reasoning) and _supports_reasoning_effort(model):
            if model in [
                "gpt-5.1",
                "gpt-5.2",
                "gpt-5.4",
                "gpt-5.4-mini",
                "gpt-5.5",
                "gpt-5.6-luna",
                "gpt-5.6-sol",
                "gpt-5.6-terra",
            ]:
                reasoning = Reasoning(effort="none")
            else:
                reasoning = Reasoning(effort="minimal")

        if client is not None and use_websocket:
            logger.warning("use_websocket is ignored when a custom client is provided, disabling")
            use_websocket = False

        self._opts = _LLMOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            reasoning=reasoning,
            service_tier=service_tier,
            verbosity=verbosity,
            max_output_tokens=max_output_tokens,
            use_websocket=use_websocket,
            prompt_cache_options=prompt_cache_options,
            prompt_cache_breakpoints=prompt_cache_breakpoints,
        )
        self._client = client
        self._owns_client = client is None
        self._ws: _ResponsesWebsocket | None = None

        self._active_streams: int = 0
        self._parallel_generation: bool = False
        self._prev_resp_id = ""
        self._prev_chat_ctx: ChatContext | None = None
        self._pending_tool_calls = set[str]()  # tool call ids that are pending for a response

        if use_websocket:
            resolved_api_key = api_key if is_given(api_key) else os.environ.get("OPENAI_API_KEY")
            if not resolved_api_key:
                raise ValueError(
                    "OpenAI API key is required, either as argument or set"
                    " OPENAI_API_KEY environment variable"
                )
            self._ws = _ResponsesWebsocket(
                api_key=resolved_api_key,
                timeout=timeout.connect if timeout is not None else None,
                model=str(model),
                base_url=base_url if is_given(base_url) else None,
            )

        else:
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

    async def _prewarm_impl(self) -> None:
        if self._ws is not None:
            self._ws._pool.prewarm()
        elif self._client is not None:
            await self._client.models.list()

    async def aclose(self) -> None:
        await super().aclose()
        if self._ws:
            await self._ws.aclose()
        if self._owns_client and self._client:
            await self._client.close()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        if self._opts.use_websocket and self._ws is not None:
            from urllib.parse import urlparse

            return urlparse(self._ws._base_url).netloc
        if self._client is not None:
            return self._client._base_url.netloc.decode("utf-8")
        return ""

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

        if is_given(self._opts.metadata):
            extra["metadata"] = self._opts.metadata

        if is_given(self._opts.user):
            extra["user"] = self._opts.user

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        if is_given(self._opts.store):
            extra["store"] = self._opts.store

        if is_given(self._opts.reasoning):
            extra["reasoning"] = self._opts.reasoning

        if is_given(self._opts.service_tier):
            extra["service_tier"] = self._opts.service_tier

        if is_given(self._opts.verbosity):
            text_cfg = extra.get("text") or {}
            extra["text"] = {**text_cfg, "verbosity": self._opts.verbosity}

        if is_given(self._opts.max_output_tokens):
            extra["max_output_tokens"] = self._opts.max_output_tokens

        if is_given(self._opts.prompt_cache_options):
            extra["prompt_cache_options"] = self._opts.prompt_cache_options

        parallel_tool_calls = (
            parallel_tool_calls if is_given(parallel_tool_calls) else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        if is_given(tool_choice):
            oai_tool_choice: response_create_params.ToolChoice
            if isinstance(tool_choice, dict):
                oai_tool_choice = {
                    "type": "function",
                    "name": tool_choice["function"]["name"],
                }
                extra["tool_choice"] = oai_tool_choice
            elif tool_choice in ("auto", "required", "none"):
                oai_tool_choice = tool_choice
                extra["tool_choice"] = oai_tool_choice

        input_chat_ctx = chat_ctx
        if (
            self._opts.store is not False
            and self._active_streams == 0
            and self._prev_chat_ctx is not None
            and self._prev_resp_id
        ):
            n = len(self._prev_chat_ctx.items)
            if ChatContext(items=chat_ctx.items[:n]).is_equivalent(
                self._prev_chat_ctx
            ) and self._pending_tool_calls_completed(chat_ctx.items[n:]):
                # send only the new items appended since the last response
                input_chat_ctx = ChatContext(items=chat_ctx.items[n:])
                extra["previous_response_id"] = self._prev_resp_id
            # if the context was modified otherwise, resend the whole context and omit previous response id
        return LLMStream(
            self,
            model=self._opts.model,
            strict_tool_schema=True,
            client=self._client if self._client else None,
            chat_ctx=input_chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
            full_chat_ctx=chat_ctx,
            prompt_cache_breakpoints=self._resolve_prompt_cache_breakpoints(),
        )

    def _resolve_prompt_cache_breakpoints(self) -> bool:
        setting = self._opts.prompt_cache_breakpoints
        if isinstance(setting, bool):
            return setting
        # subclasses (xAI) and custom base URLs reach hosts not verified to accept the field
        return self._host() == _OPENAI_HOST and supports_prompt_cache_breakpoints(
            str(self._opts.model)
        )

    def _host(self) -> str:
        # the hostname, not the netloc: an explicit port or a trailing dot is still OpenAI
        if self._opts.use_websocket and self._ws is not None:
            host = URL(self._ws._base_url).host
        elif self._client is not None:
            host = self._client._base_url.host
        else:
            host = None
        return (host or "").rstrip(".").lower()

    def _pending_tool_calls_completed(self, items: list[ChatItem]) -> bool:
        if not self._pending_tool_calls:
            return True

        completed_tool_calls = {
            item.call_id for item in items if item.type == "function_call_output"
        }
        return all(call_id in completed_tool_calls for call_id in self._pending_tool_calls)
