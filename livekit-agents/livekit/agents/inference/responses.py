"""Responses API LLM served by the LiveKit inference gateway."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import aiohttp
import httpx2 as httpx
import openai
from openai.types import Reasoning
from openai.types.responses import (
    ResponseErrorEvent,
    ResponseIncludable,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    response_create_params,
)
from openai.types.responses.response_create_params import PromptCacheOptions
from openai.types.shared_params import Metadata
from typing_extensions import TypedDict
from yarl import URL

from .. import llm
from .._exceptions import APIConnectionError, APIStatusError, APITimeoutError
from ..llm import ToolChoice, utils as llm_utils
from ..llm._responses_engine import (
    _WS_HEARTBEAT,
    ResponsesLLMStream,
    ResponsesWebsocket,
)
from ..llm.chat_context import ChatContext
from ..llm.tool_context import Tool
from ..log import logger
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGiven,
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
from .llm import LLMModels, supports_prompt_cache_breakpoints

# the gateway closes a socket 60 minutes after the upgrade (WSConnectionMaxLifetime) and the
# pool checks age only at get(), so leave room for a turn to finish before that
_GATEWAY_WS_MAX_SESSION = 50 * 60

# status and retryability per gateway.error code; agent-gateway's GatewayErrCode* constants.
# upstream_timeout (the gateway's upstream stall bound, agent-gateway#1823) is not here:
# _handle_error raises it as a retryable APITimeoutError instead of a status error
_GATEWAY_ERROR_STATUS: dict[str, tuple[int, bool | None]] = {
    "upstream_connect_failed": (502, True),
    "quota_exceeded": (429, None),
    "model_mismatch": (400, False),
    "invalid_event": (400, False),
    "tool_not_allowed": (400, False),  # hosted tools other than web search (agent-gateway#1790)
    "unknown_model": (400, False),
    "no_ws_deployments": (400, False),
    "no_ws_tag_match": (400, False),
    "no_hipaa_compliant_deployment": (400, False),
    "engine_unsupported": (400, False),
    "moderation_blocked": (400, False),
    "in_flight_response": (400, False),
}


class ResponsesOptions(TypedDict, total=False):
    include: list[ResponseIncludable] | None
    instructions: str | None
    max_output_tokens: int | None
    max_tool_calls: int | None
    metadata: Metadata | None
    parallel_tool_calls: bool | None
    prompt_cache_options: PromptCacheOptions
    reasoning: Reasoning | dict[str, Any] | None
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
    """Responses API LLM routed through the LiveKit inference gateway.

    With ``use_websocket=True`` turns stream over ``/v1/responses`` on pooled sockets, so
    ``inference_class`` and the other upgrade headers are fixed per connection.

    Each read waits ``conn_options.timeout`` for the next event. REST streams get gateway
    keepalives while a model reasons silently; over WebSocket they are rolling out behind a
    gateway flag, so until then raise ``conn_options.timeout`` for models that reason silently
    for longer than it.
    """

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
        use_websocket: bool = False,
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
            use_websocket=use_websocket,
        )
        self._ws: ResponsesWebsocket | None = None
        self._retired_ws: set[_GatewayResponsesWebsocket] = set()
        self._ws_close_tasks: set[asyncio.Task[None]] = set()
        # the gateway bounds upstream silence, so reads only need to outlast its keepalives
        self._stream_idle_timeout: float | None = None
        self._active_streams: int = 0
        self._parallel_generation: bool = False
        self._prev_resp_id = ""
        self._prev_chat_ctx: ChatContext | None = None
        self._pending_tool_calls = set[str]()
        self._client: openai.AsyncClient | None = None
        if use_websocket:
            self._ws = self._make_ws()
        else:
            self._client = self._make_client()

    def _make_ws(self) -> _GatewayResponsesWebsocket:
        return _GatewayResponsesWebsocket(
            base_url=self._opts.base_url,
            model=self._opts.model,
            api_key=self._opts.api_key,
            api_secret=self._opts.api_secret,
            provider=self._opts.provider,
            inference_class=self._opts.inference_class,
            extra_headers=lambda: dict(self._opts.extra_kwargs.get("extra_headers") or {}),
            timeout=None,
        )

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
        if self._ws is not None:
            self._ws._pool.prewarm()
        elif self._client is not None:
            await self._client.models.list()

    async def aclose(self) -> None:
        await super().aclose()
        if self._ws is not None:
            await self._ws.aclose()
        for ws in self._retired_ws:
            await ws.aclose()
        self._retired_ws.clear()
        if self._ws_close_tasks:
            await asyncio.gather(*self._ws_close_tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.close()

    def update_options(
        self,
        *,
        model: NotGivenOr[LLMModels | str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[ResponsesOptions | dict[str, Any]] = NOT_GIVEN,
        prompt_cache_breakpoints: NotGivenOr[bool | Literal["auto"]] = NOT_GIVEN,
    ) -> None:
        """Update LLM configuration options.

        Each option is read on the next ``chat()`` call. ``extra_kwargs`` *replaces* the
        persistent kwargs dict rather than merging; pass ``{}`` to clear it. Over WebSocket a
        new ``model`` gets a new pool and new ``extra_headers`` invalidate it, since both are
        bound to the socket at the upgrade. A turn created before a model swap keeps the old
        model's pool, so its retries dial the model it sends.
        """
        if is_given(model) and model != self._opts.model:
            self._opts.model = model
            if isinstance(self._ws, _GatewayResponsesWebsocket):
                self._retire_ws(self._ws)
                self._ws = self._make_ws()
        if is_given(extra_kwargs):
            old_headers = self._opts.extra_kwargs.get("extra_headers")
            self._opts.extra_kwargs = dict(extra_kwargs)
            # headers ride on the upgrade, so pooled sockets carry the old ones
            if self._ws is not None and extra_kwargs.get("extra_headers") != old_headers:
                self._ws._pool.invalidate()
        # isinstance rather than is_given: mypy cannot narrow the bool | Literal union through it
        if not isinstance(prompt_cache_breakpoints, NotGiven):
            self._opts.prompt_cache_breakpoints = prompt_cache_breakpoints

    def _retire_ws(self, ws: _GatewayResponsesWebsocket) -> None:
        ws._pool.invalidate()
        self._retired_ws.add(ws)
        self._close_retired_ws(ws)

    def _release_ws(self, ws: _GatewayResponsesWebsocket) -> None:
        ws._holders -= 1
        self._close_retired_ws(ws)

    def _close_retired_ws(self, ws: _GatewayResponsesWebsocket) -> None:
        if ws not in self._retired_ws or ws._holders > 0:
            return
        try:
            task = asyncio.get_running_loop().create_task(ws.aclose())
        except RuntimeError:
            return  # no loop to close on; aclose() closes it
        self._retired_ws.discard(ws)
        self._ws_close_tasks.add(task)
        task.add_done_callback(self._ws_close_tasks.discard)

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
        response_format: NotGivenOr[type[llm_utils.ResponseFormatT] | dict[str, Any]] = NOT_GIVEN,
        inference_class: NotGivenOr[InferenceClass] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> ResponsesLLMStream:
        extra: dict[str, Any] = dict(self._opts.extra_kwargs)
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(response_format):
            text = dict(extra.get("text") or {})
            text["format"] = _to_text_format(response_format)
            extra["text"] = text

        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        merged_tool_choice = extra.pop("tool_choice", None)
        oai_tool_choice = _to_responses_tool_choice(
            tool_choice if is_given(tool_choice) else merged_tool_choice
        )
        # the API rejects a tool_choice with no tools
        if oai_tool_choice is not None and tools:
            extra["tool_choice"] = oai_tool_choice

        if self._client is not None:
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
        else:
            # headers ride on the upgrade, so only the LLM's own extra_headers reach the
            # gateway; the payload must not carry them
            extra.pop("extra_headers", None)
            if is_given(extra_kwargs) and "extra_headers" in extra_kwargs:
                logger.warning(
                    "extra_headers cannot change per request over WebSocket; "
                    "using the LLM's extra_kwargs headers"
                )
            if is_given(inference_class) and inference_class != self._opts.inference_class:
                logger.warning(
                    "inference_class cannot change per request over WebSocket; "
                    "using the LLM's value",
                    extra={
                        "requested": inference_class,
                        "configured": self._opts.inference_class,
                    },
                )

        stream = _GatewayResponsesLLMStream(
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
        if isinstance(self._ws, _GatewayResponsesWebsocket):
            ws = self._ws
            ws._holders += 1
            stream._task.add_done_callback(lambda _: self._release_ws(ws))
        return stream

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
        return cast(response_create_params.ToolChoice, tool_choice)
    return None


def _to_text_format(response_format: Any) -> dict[str, Any]:
    if isinstance(response_format, dict):
        return _dict_to_text_format(response_format)
    chat_format = llm_utils.to_openai_response_format(response_format)
    schema = chat_format["json_schema"]
    return {
        "type": "json_schema",
        "name": schema["name"],
        "schema": schema["schema"],
        "strict": schema["strict"],
    }


def _dict_to_text_format(response_format: dict[str, Any]) -> dict[str, Any]:
    # accepts the Chat Completions shape, with the schema nested under "json_schema", or
    # a dict already in the Responses text.format shape
    fmt_type = response_format.get("type")
    if fmt_type in ("text", "json_object"):
        return dict(response_format)
    if fmt_type != "json_schema":
        raise TypeError(f"unsupported response_format type: {fmt_type!r}")
    schema = response_format.get("json_schema", response_format)
    if not isinstance(schema, dict) or "name" not in schema or "schema" not in schema:
        raise TypeError("a json_schema response_format needs a name and a schema")
    text_format = {"type": "json_schema", "name": schema["name"], "schema": schema["schema"]}
    for key in ("strict", "description"):
        if key in schema:
            text_format[key] = schema[key]
    return text_format


class _GatewayResponsesWebsocket(ResponsesWebsocket):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        api_secret: str,
        provider: str | None,
        inference_class: InferenceClass | None,
        extra_headers: Callable[[], dict[str, str]],
        timeout: float | None,
    ) -> None:
        super().__init__(
            api_key="",
            timeout=timeout,
            model=model,
            base_url=f"{base_url.rstrip('/')}/responses",
            max_session_duration=_GATEWAY_WS_MAX_SESSION,
        )
        self._lk_api_key = api_key
        self._lk_api_secret = api_secret
        self._provider = provider
        self._inference_class = inference_class
        self._extra_headers = extra_headers
        # streams bound to this transport that have not finished
        self._holders = 0

    async def _create_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        token = create_access_token(self._lk_api_key, self._lk_api_secret)
        headers = {
            **self._extra_headers(),
            **get_inference_headers(inference_class=self._inference_class),
            "Authorization": f"Bearer {token}",
        }
        if self._provider:
            headers[HEADER_INFERENCE_PROVIDER] = self._provider
        try:
            return await asyncio.wait_for(
                self._ensure_http_session().ws_connect(
                    self._base_url, headers=headers, heartbeat=_WS_HEARTBEAT
                ),
                timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            if e.status == 429:
                self._log_rate_limited(e)
            raise APIStatusError(
                f"inference gateway refused the WebSocket: {e.message}", status_code=e.status
            ) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError("failed to connect to inference gateway WebSocket") from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError("timed out connecting to inference gateway WebSocket") from e

    def _log_rate_limited(self, e: aiohttp.WSServerHandshakeError) -> None:
        extra: dict[str, Any] = {"model": URL(self._base_url).query.get("model")}
        extra.update(extract_quota_usage(e.headers or {}))
        logger.warning("LLM request rate limited by inference gateway", extra=extra)


class _GatewayResponsesLLMStream(ResponsesLLMStream):
    def _parse_ws_event(self, event: dict) -> ResponseStreamEvent | None:
        event_type = event.get("type")
        if event_type == "gateway.error":
            return ResponseErrorEvent.model_validate(
                {
                    "type": "error",
                    "sequence_number": -1,
                    "code": event.get("code"),
                    "message": event.get("message", "gateway error"),
                    "param": None,
                }
            )
        if event_type in ("gateway.connected", "gateway.session_expiring", "gateway.keepalive"):
            return None
        return super()._parse_ws_event(event)

    def _handle_error(self, event: ResponseErrorEvent) -> None:
        if event.code == "upstream_timeout":
            raise APITimeoutError(f"{event.message} ({event.code})", retryable=True)
        mapped = _GATEWAY_ERROR_STATUS.get(event.code or "")
        if mapped is None:
            super()._handle_error(event)
            return
        status_code, retryable = mapped
        raise APIStatusError(
            f"{event.message} ({event.code})", status_code=status_code, retryable=retryable
        )

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
