"""Responses API LLM served by the LiveKit inference gateway."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx2 as httpx
import openai
from openai.types import Reasoning
from openai.types.responses import (
    ResponseIncludable,
    ResponseTextConfigParam,
    response_create_params,
)
from openai.types.shared_params import Metadata
from typing_extensions import TypedDict

from .. import llm
from .._exceptions import APIStatusError
from ..llm import ToolChoice
from ..llm._responses_engine import ResponsesLLMStream, ResponsesWebsocket
from ..llm.chat_context import ChatContext
from ..llm.tool_context import Tool
from ..log import logger
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import is_given
from ._utils import (
    HEADER_INFERENCE_PROVIDER,
    InferenceClass,
    create_access_token,
    extract_quota_usage,
    get_default_inference_url,
    get_inference_headers,
    resolve_credentials,
)
from .llm import LLMModels, PromptCacheOptions, supports_prompt_cache_breakpoints


class ResponsesOptions(TypedDict, total=False):
    include: list[ResponseIncludable] | None
    instructions: str | None
    max_output_tokens: int | None
    max_tool_calls: int | None
    metadata: Metadata | None
    parallel_tool_calls: bool | None
    prompt_cache_options: PromptCacheOptions
    reasoning: Reasoning | dict[str, Any] | None
    service_tier: Literal["auto", "default", "flex", "scale", "priority", "ultrafast"] | None
    temperature: float | None
    text: ResponseTextConfigParam
    top_p: float | None
    truncation: Literal["auto", "disabled"] | None
    user: str

    # livekit-typed argument
    tool_choice: ToolChoice


@dataclass
class _LLMOptions:
    model: LLMModels | str
    provider: str | None
    base_url: str
    api_key: str
    api_secret: str
    inference_class: InferenceClass | None
    extra_kwargs: ResponsesOptions | dict[str, Any]
    prompt_cache_breakpoints: bool | Literal["auto"]
    use_websocket: bool = False


class LLM(llm.LLM):
    """Responses API LLM routed through the LiveKit inference gateway."""

    _provider_tool_type: type[llm.ProviderTool] = llm.ProviderTool

    def __init__(
        self,
        model: LLMModels | str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        inference_class: InferenceClass | None = None,
        extra_kwargs: ResponsesOptions | dict[str, Any] | None = None,
        prompt_cache_breakpoints: bool | Literal["auto"] = "auto",
    ) -> None:
        super().__init__()

        lk_api_key, lk_api_secret = resolve_credentials(api_key, api_secret)
        self._opts = _LLMOptions(
            model=model,
            provider=provider,
            base_url=base_url if base_url else get_default_inference_url(),
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            inference_class=inference_class,
            extra_kwargs=extra_kwargs or {},
            prompt_cache_breakpoints=prompt_cache_breakpoints,
        )
        self._ws: ResponsesWebsocket | None = None
        self._active_streams: int = 0
        self._parallel_generation: bool = False
        self._prev_resp_id = ""
        self._prev_chat_ctx: ChatContext | None = None
        self._pending_tool_calls = set[str]()
        self._client = self._make_client()

    def _make_client(self) -> openai.AsyncClient:
        return openai.AsyncClient(
            api_key=create_access_token(self._opts.api_key, self._opts.api_secret),
            base_url=self._opts.base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
                ),
            ),
        )

    async def _prewarm_impl(self) -> None:
        await self._client.models.list()

    async def aclose(self) -> None:
        await super().aclose()
        await self._client.close()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "livekit"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        inference_class: NotGivenOr[InferenceClass] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> ResponsesLLMStream:
        extra: dict[str, Any] = dict(self._opts.extra_kwargs)
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        merged_tool_choice = extra.pop("tool_choice", None)
        oai_tool_choice = _to_responses_tool_choice(
            tool_choice if is_given(tool_choice) else merged_tool_choice
        )
        if oai_tool_choice is not None:
            extra["tool_choice"] = oai_tool_choice

        effective_inference_class = (
            inference_class if is_given(inference_class) else self._opts.inference_class
        )
        headers = {
            **extra.get("extra_headers", {}),
            **get_inference_headers(inference_class=effective_inference_class),
        }
        if self._opts.provider:
            headers[HEADER_INFERENCE_PROVIDER] = self._opts.provider
        extra["extra_headers"] = headers

        self._client.api_key = create_access_token(self._opts.api_key, self._opts.api_secret)
        return _GatewayResponsesLLMStream(
            self,
            model=self._opts.model,
            strict_tool_schema=True,
            client=self._client,
            chat_ctx=chat_ctx,
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
        return supports_prompt_cache_breakpoints(self._opts.model)


def _to_responses_tool_choice(tool_choice: Any) -> response_create_params.ToolChoice | None:
    # livekit names a function as {"type": "function", "function": {"name": ...}};
    # a dict already in the Responses shape passes through
    if isinstance(tool_choice, dict) and "function" in tool_choice:
        return {"type": "function", "name": tool_choice["function"]["name"]}
    if isinstance(tool_choice, dict) or tool_choice in ("auto", "required", "none"):
        return tool_choice
    return None


class _GatewayResponsesLLMStream(ResponsesLLMStream):
    async def _run_impl(self) -> None:
        try:
            await super()._run_impl()
        except APIStatusError as e:
            # the engine raises without `from`, so __context__ holds the openai error
            if e.status_code == 429 and isinstance(e.__context__, openai.APIStatusError):
                self._log_rate_limited(e.__context__)
            raise

    def _log_rate_limited(self, e: openai.APIStatusError) -> None:
        extra: dict[str, Any] = {"model": self._model, "request_id": e.request_id}
        extra.update(extract_quota_usage(e.response.headers))
        logger.warning("LLM request rate limited by inference gateway", extra=extra)
