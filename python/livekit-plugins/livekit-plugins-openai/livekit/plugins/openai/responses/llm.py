from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal, TypedDict
from urllib.parse import urlparse

import httpx2 as httpx
from pydantic import JsonValue, TypeAdapter, ValidationError
from yarl import URL

import openai
from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    TokenUsage,
    http,
    llm,
    utils,
)
from livekit.agents.inference.llm import drop_unsupported_params
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import (
    Tool,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGiven,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types import Reasoning
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseTextConfig,
    ResponseTextDeltaEvent,
    response_create_params,
)
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.shared_params import Reasoning as ReasoningParam, ResponsesModel

from ..log import logger
from ..models import supports_reasoning_effort
from ..tools import OpenAITool
from ..utils import error_kind, http_timeout

ServiceTier = Literal["auto", "default", "flex", "scale", "priority", "ultrafast"]
Verbosity = Literal["low", "medium", "high"]

OPENAI_RESPONSES_WS_URL = "wss://api.openai.com/v1/responses"

# ws ping interval; keeps idle pooled sockets warm and lets the client detect dead peers
_WS_HEARTBEAT = 30.0


class _ResponseCreate(TypedDict):
    """A ``response.create`` frame's own fields; the request's params go beside them. The
    framework's tool schemas include provider tools the SDK's types do not name."""

    type: Literal["response.create"]
    tools: list[dict[str, object]]


class _ErrorBody(TypedDict, total=False):
    code: str | None
    type: str | None
    message: str


class _ErrorFrame(TypedDict, total=False):
    """An ``error`` frame: the request failed without a response. OpenAI nests the error under
    ``error`` beside the HTTP ``status``; gateways put its fields at the top level."""

    status: int
    error: _ErrorBody
    code: str | None
    message: str


_FRAME = TypeAdapter(dict[str, JsonValue])
_ERROR_FRAME = TypeAdapter(_ErrorFrame)
_REASONING = TypeAdapter(ReasoningParam)
# the events the stream reads; it skips the others, since the API adds event types
_EVENTS: dict[
    str,
    type[ResponseCreatedEvent]
    | type[ResponseOutputItemDoneEvent]
    | type[ResponseTextDeltaEvent]
    | type[ResponseCompletedEvent]
    | type[ResponseFailedEvent]
    | type[ResponseIncompleteEvent],
] = {
    "response.created": ResponseCreatedEvent,
    "response.output_item.done": ResponseOutputItemDoneEvent,
    "response.output_text.delta": ResponseTextDeltaEvent,
    "response.completed": ResponseCompletedEvent,
    "response.failed": ResponseFailedEvent,
    "response.incomplete": ResponseIncompleteEvent,
}


def _json_default(o: object) -> object:
    if isinstance(o, openai.BaseModel):
        # a caller's field given as an SDK model goes as the SDK sends it over HTTP: only the
        # fields set, since the API rejects an explicit null on an enum field
        return o.model_dump(mode="json", exclude_unset=True, by_alias=True)
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def _decode(message: http.WebSocketMessage) -> ResponseStreamEvent | None:
    """A received frame as the event the stream reads, ``None`` for one it does not read."""
    if message.type is not http.WebSocketMessageType.TEXT:
        raise APIError(
            "OpenAI Responses sent a binary frame", kind="protocol", detail=repr(message.data)
        )
    text = message.text
    try:
        frame = _FRAME.validate_json(text)
        tag = frame.get("type")
        if tag == "error":
            raise _frame_error(_ERROR_FRAME.validate_python(frame))
        event = _EVENTS.get(tag) if isinstance(tag, str) else None
        return event.model_validate(frame) if event is not None else None
    except ValidationError:
        raise APIError(
            "OpenAI Responses sent a malformed event", kind="protocol", detail=text
        ) from None


def _frame_error(frame: _ErrorFrame) -> APIError:
    body = frame.get("error", {})
    message = body.get("message") or frame.get("message") or "OpenAI Responses failed the request"
    code = body.get("code") or frame.get("code") or body.get("type")
    status = frame.get("status")
    if status is None and code is not None and code.isdigit():
        status = int(code)
    if status is None:
        return APIError(message, kind=error_kind(code))
    # what OpenAI's code says failed, when it says, else what the status says
    kind = error_kind(code)
    return APIStatusError(
        message, status_code=status, kind=kind if kind != "invalid_request" else None
    )


async def _still_open(ws: http.WebSocket) -> bool:
    """Whether an idle socket is still open: one the peer closed while idle has its close
    already queued, and any other frame is one no request is owed."""
    if ws.closed:
        return False
    try:
        await ws.receive(timeout=0)
    except APITimeoutError:
        return True
    except APIConnectionError:
        return False
    return False


class _ResponsesWebsocket:
    """Sockets for the Responses API's WebSocket mode, each carrying one response at a time."""

    def __init__(
        self, api_key: str, timeout: float | None, model: str, base_url: str | None = None
    ) -> None:
        self._api_key = api_key
        url = URL(base_url if base_url else OPENAI_RESPONSES_WS_URL)
        if url.scheme in ("http", "https"):
            url = url.with_scheme("ws" if url.scheme == "http" else "wss")
        if url.host != "api.openai.com":
            # OpenAI's native endpoint takes the model in the response.create
            # payload; gateways need it on the upgrade URL to route the
            # connection before the first frame.
            url = url.update_query(model=model)
        self.url = url
        self._session: http.HttpSession | None = None
        self._pool = utils.ConnectionPool[http.WebSocket](
            connect_cb=self._connect,
            close_cb=self._close,
            max_session_duration=3600,
            conn_options=APIConnectOptions(
                connect_timeout=(
                    timeout if timeout is not None else DEFAULT_API_CONNECT_OPTIONS.connect_timeout
                )
            ),
        )

    async def _connect(self, conn_options: APIConnectOptions) -> http.WebSocket:
        if self._session is None:
            self._session = http.http_session()
        return await self._session.ws_connect(
            str(self.url),
            headers={"Authorization": f"Bearer {self._api_key}"},
            heartbeat=_WS_HEARTBEAT,
            conn_options=conn_options,
        )

    async def _close(self, ws: http.WebSocket) -> None:
        await ws.close()

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        await self._pool.aclose()

    @asynccontextmanager
    async def exchange(
        self, request: str, conn_options: APIConnectOptions
    ) -> AsyncIterator[http.WebSocket]:
        """A socket that carried ``request``, for its response: back to the pool when the block
        exits (the response completed), closed when it raises."""
        ws = await self._open_socket(conn_options)
        try:
            await ws.send_str(request)
            yield ws
        except BaseException:
            self._pool.remove(ws)
            raise
        self._pool.put(ws)

    async def _open_socket(self, conn_options: APIConnectOptions) -> http.WebSocket:
        # a pooled socket the peer closed while it sat idle is replaced before the request
        # goes, so no retry is spent on it; a new socket ends the search
        while True:
            ws = await self._pool.get(conn_options=conn_options)
            if not self._pool.last_connection_reused or await _still_open(ws):
                return ws
            self._pool.remove(ws)


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


@dataclass
class _ResponseCache:
    _active_streams: int = 0
    _parallel_generation: bool = False
    _response_id: str = ""
    _chat_ctx: ChatContext | None = None
    _pending_tool_calls: set[str] = field(default_factory=set[str])

    def prepare(self, chat_ctx: ChatContext) -> tuple[ChatContext, str | None]:
        if self._active_streams or self._chat_ctx is None or not self._response_id:
            return chat_ctx, None
        n = len(self._chat_ctx.items)
        if not ChatContext(items=chat_ctx.items[:n]).is_equivalent(self._chat_ctx):
            return chat_ctx, None
        completed = {
            item.call_id for item in chat_ctx.items[n:] if item.type == "function_call_output"
        }
        if not self._pending_tool_calls.issubset(completed):
            return chat_ctx, None
        return ChatContext(items=chat_ctx.items[n:]), self._response_id

    def begin(self) -> None:
        self._parallel_generation |= self._active_streams > 0
        self._active_streams += 1

    def end(self) -> None:
        self._active_streams -= 1
        if self._active_streams == 0 and self._parallel_generation:
            self._response_id = ""
            self._chat_ctx = None
            self._pending_tool_calls.clear()
            self._parallel_generation = False

    def complete(
        self, response_id: str, chat_ctx: ChatContext, pending_tool_calls: set[str]
    ) -> None:
        self._response_id = response_id
        self._chat_ctx = chat_ctx
        self._pending_tool_calls = set(pending_tool_calls)


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
    ) -> None:
        """
        Create a new instance of OpenAI Responses LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """
        super().__init__()

        if not is_given(reasoning) and supports_reasoning_effort(model):
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
        )
        self._client = client
        self._owns_client = client is None
        self._ws: _ResponsesWebsocket | None = None

        self._response_cache = _ResponseCache()

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
            self._ws.prewarm()
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
        if self._ws is not None:
            return urlparse(str(self._ws.url)).netloc
        if self._client is not None:
            return self._client.base_url.netloc.decode("utf-8")
        return ""

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, object]] = NOT_GIVEN,
    ) -> LLMStream:
        opts = self._opts
        # the caller's own fields; an option this model sets replaces one of the same name
        extra: dict[str, object] = dict(extra_kwargs) if is_given(extra_kwargs) else {}
        params: response_create_params.ResponseCreateParamsBase = {}
        if is_given(opts.metadata):
            params["metadata"] = opts.metadata
        if is_given(opts.user):
            params["user"] = opts.user
        if is_given(opts.temperature):
            params["temperature"] = opts.temperature
        if is_given(opts.store):
            params["store"] = opts.store
        if is_given(opts.reasoning):
            # unset fields are left out: the API rejects an explicit null on an enum field
            params["reasoning"] = _REASONING.validate_python(
                opts.reasoning.model_dump(mode="json", exclude_none=True)
            )
        if is_given(opts.service_tier):
            params["service_tier"] = opts.service_tier
        if is_given(opts.max_output_tokens):
            params["max_output_tokens"] = opts.max_output_tokens

        parallel_tool_calls = (
            parallel_tool_calls if is_given(parallel_tool_calls) else opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            params["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if not isinstance(tool_choice, NotGiven) else opts.tool_choice
        if not isinstance(tool_choice, NotGiven):
            if not isinstance(tool_choice, str):
                params["tool_choice"] = {
                    "type": "function",
                    "name": tool_choice["function"]["name"],
                }
            elif tool_choice in ("auto", "required", "none"):
                params["tool_choice"] = tool_choice

        if is_given(opts.verbosity):
            # verbosity joins the caller's text format, which keeps its wire field names
            text_cfg = ResponseTextConfig.model_validate(extra.get("text") or {})
            text_cfg.verbosity = opts.verbosity
            extra["text"] = text_cfg.model_dump(by_alias=True, exclude_unset=True)

        input_chat_ctx = chat_ctx
        if opts.store is not False:
            input_chat_ctx, previous_response_id = self._response_cache.prepare(chat_ctx)
            if previous_response_id is not None:
                params["previous_response_id"] = previous_response_id

        # what the model refuses at its reasoning effort is left out; of the options, that can
        # only be the temperature
        view: dict[str, object] = {**extra, **params}
        if is_given(opts.reasoning):
            view["reasoning"] = opts.reasoning
        kept = drop_unsupported_params(opts.model, view)
        if "temperature" not in kept:
            params.pop("temperature", None)
        extra = {k: v for k, v in extra.items() if k in kept and k not in params}

        return LLMStream(
            self,
            model=opts.model,
            strict_tool_schema=True,
            client=self._client,
            chat_ctx=input_chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            params=params,
            extra=extra,
            full_chat_ctx=chat_ctx,
            websocket=self._ws,
            cache=self._response_cache,
            provider_tool_type=self._provider_tool_type,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        model: str | ResponsesModel,
        strict_tool_schema: bool,
        client: openai.AsyncClient | None,
        chat_ctx: llm.ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        params: response_create_params.ResponseCreateParamsBase,
        extra: dict[str, object],
        full_chat_ctx: llm.ChatContext,
        websocket: _ResponsesWebsocket | None,
        cache: _ResponseCache,
        provider_tool_type: type[llm.ProviderTool],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._strict_tool_schema = strict_tool_schema
        self._response_id: str = ""
        self._response_completed: bool = False
        self._pending_tool_calls = set[str]()

        self._client = client
        self._websocket = websocket
        self._cache = cache
        self._provider_tool_type = provider_tool_type
        self._params = params
        self._extra = extra
        self._full_chat_ctx = full_chat_ctx.copy()

    async def _run(self) -> None:
        self._cache.begin()
        try:
            await self._run_impl()
        finally:
            self._cache.end()

    async def _run_impl(self) -> None:
        self._response_completed = False
        chat_ctx, _ = await self._chat_ctx.to_provider_format(format="openai.responses")
        self._tool_ctx = llm.ToolContext(self.tools)
        tool_schemas = self._tool_ctx.parse_function_tools(
            "openai.responses",
            strict=self._strict_tool_schema,
            provider_tool_type=self._provider_tool_type,
        )

        request = self._params.copy()
        request["model"] = self._model
        request["input"] = chat_ctx

        if self._websocket is not None:
            create: _ResponseCreate = {"type": "response.create", "tools": tool_schemas}
            frame = json.dumps({**create, **request, **self._extra}, default=_json_default)
            await self._run_websocket(self._websocket, frame)
            return

        assert self._client is not None
        try:
            stream = await self._client.with_options(max_retries=0).responses.create(
                **request,
                stream=True,
                timeout=http_timeout(self._conn_options),
                # the framework's tool schemas include provider tools the SDK's types do not name
                extra_body={"tools": tool_schemas, **self._extra},
            )
            async with stream:
                async for event in stream:
                    self._process_event(event)
        # only what the SDK reports of the transport and the API; anything else is a bug
        except openai.APITimeoutError as e:
            raise APITimeoutError() from e
        except openai.APIConnectionError as e:
            raise APIConnectionError() from e
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message, status_code=e.status_code, request_id=e.request_id, body=e.body
            ) from None
        except openai.APIResponseValidationError as e:
            raise APIError(
                "OpenAI Responses sent a malformed response", kind="protocol", detail=e.message
            ) from None
        except openai.APIError as e:
            # an error the API reported mid-stream
            raise APIError(e.message, kind=error_kind(e.code or e.type), body=e.body) from None

    async def _run_websocket(self, websocket: _ResponsesWebsocket, request: str) -> None:
        attempts = self._core_attempts
        async with websocket.exchange(request, self._conn_options) as ws:
            attempts.expect_response()
            while not self._response_completed:
                message = await ws.receive()
                if message.type is http.WebSocketMessageType.CLOSE:
                    raise ws.close_error("OpenAI Responses closed the connection mid-response")
                attempts.progress()
                self._process_event(_decode(message))
            attempts.complete_response()

    def _process_event(self, event: ResponseStreamEvent | None) -> llm.ChatChunk | None:
        """Handle one stream event, returning the chunk it sent to the caller, if any."""
        if event is None:
            return None
        chunk = None
        if isinstance(event, ResponseErrorEvent):
            self._handle_error(event)
        if isinstance(event, ResponseCreatedEvent):
            self._handle_response_created(event)
        if isinstance(event, ResponseOutputItemDoneEvent):
            chunk = self._handle_output_items_done(event)
        if isinstance(event, ResponseTextDeltaEvent):
            chunk = self._handle_response_output_text_delta(event)
        if isinstance(event, ResponseCompletedEvent):
            chunk = self._handle_response_completed(event)
        if isinstance(event, ResponseFailedEvent):
            self._handle_response_failed(event)
        if isinstance(event, ResponseIncompleteEvent):
            self._handle_response_incomplete(event)
        if chunk is not None:
            self.send(chunk)
        return chunk

    def _handle_error(self, event: ResponseErrorEvent) -> None:
        if event.code and event.code.isdigit():
            raise APIStatusError(event.message, status_code=int(event.code))
        raise APIError(event.message, kind=error_kind(event.code))

    def _handle_response_failed(self, event: ResponseFailedEvent) -> None:
        err = event.response.error
        raise APIError(
            err.message if err else "response.failed", kind=error_kind(err.code if err else None)
        )

    def _handle_response_incomplete(self, event: ResponseIncompleteEvent) -> None:
        details = event.response.incomplete_details
        reason = details.reason if details else None
        raise APIError(
            f"response incomplete: {reason or 'reason unavailable'}", kind="invalid_request"
        )

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        self._response_id = event.response.id

    def _handle_response_completed(self, event: ResponseCompletedEvent) -> llm.ChatChunk | None:
        for item in event.response.output:
            # Every item.type is a discriminator of openai's ResponseOutputItem union.
            # Of those, only these are produced/consumed by the agent itself; all other
            # members of the union are tools the Responses API runs server-side (e.g.
            # openai web_search, xAI web_search and x_search's custom_tool_call subcalls),
            # so anything not in this set is a provider-executed tool.
            if item.type not in ("message", "reasoning", "function_call", "function_call_output"):
                logger.info(
                    "provider tool executed",
                    extra={
                        "tool_type": item.type,
                        "lk.pii.result": item.model_dump(exclude_none=True),
                    },
                )

        self._response_completed = True
        self._cache.complete(self._response_id, self._full_chat_ctx, self._pending_tool_calls)

        chunk = None
        if usage := event.response.usage:
            chunk = llm.ChatChunk(
                id=self._response_id,
                usage=TokenUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.input_tokens_details.cached_tokens
                    if usage.input_tokens_details
                    else None,
                    output_reasoning_tokens=usage.output_tokens_details.reasoning_tokens
                    if usage.output_tokens_details
                    else None,
                ),
            )
        return chunk

    def _handle_output_items_done(self, event: ResponseOutputItemDoneEvent) -> llm.ChatChunk | None:
        chunk = None
        if event.item.type == "function_call":
            chunk = llm.ChatChunk(
                id=self._response_id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        llm.FunctionToolCall(
                            arguments=event.item.arguments,
                            name=event.item.name,
                            call_id=event.item.call_id,
                        )
                    ],
                ),
            )
            self._pending_tool_calls.add(event.item.call_id)
        elif isinstance(event.item, ResponseOutputMessage) and event.item.phase is not None:
            # Models like gpt-5.3-codex label assistant messages as intermediate
            # `commentary` or the `final_answer`
            chunk = llm.ChatChunk(
                id=self._response_id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content=None,
                    extra={"openai": {"phase": event.item.phase}},
                ),
            )
        return chunk

    def _handle_response_output_text_delta(
        self, event: ResponseTextDeltaEvent
    ) -> llm.ChatChunk | None:
        return llm.ChatChunk(
            id=self._response_id,
            delta=llm.ChoiceDelta(content=event.delta, role="assistant"),
        )
