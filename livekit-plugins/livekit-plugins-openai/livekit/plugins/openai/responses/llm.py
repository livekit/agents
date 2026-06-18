from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal, cast

import aiohttp
import httpx

import openai
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm, utils
from livekit.agents.inference.llm import drop_unsupported_params
from livekit.agents.llm import ToolChoice
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
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseInputParam,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ToolParam,
    response_create_params,
)
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.shared_params import ResponsesModel

from ..log import logger
from ..models import _supports_reasoning_effort

OPENAI_RESPONSES_WS_URL = "wss://api.openai.com/v1/responses"


class _ResponsesWebsocket:
    def __init__(
        self, api_key: str | None, timeout: float | None, base_url: str | None = None
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout or DEFAULT_API_CONNECT_OPTIONS.timeout
        self._base_url = base_url if base_url else OPENAI_RESPONSES_WS_URL

        self._session: aiohttp.ClientSession | None = None

        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._create_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,
        )

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def _create_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        try:
            return await asyncio.wait_for(
                self._ensure_http_session().ws_connect(
                    self._base_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                ),
                timeout,
            )
        except aiohttp.ClientError as e:
            raise APIConnectionError("failed to connect to OpenAI Responses WebSocket") from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError("timed out connecting to OpenAI Responses WebSocket") from e

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    async def aclose(self) -> None:
        await self._pool.aclose()

    async def generate_response(self, msg: dict) -> AsyncGenerator[dict, None]:
        def _default(o: object) -> object:
            if isinstance(o, openai.BaseModel):
                return o.model_dump(mode="json")
            raise TypeError(f"unexpected type {type(o)}")

        try:
            data = json.dumps(msg, default=_default)
        except TypeError as e:
            raise APIConnectionError(f"failed to serialize request: {e}") from e

        async with self._pool.connection(timeout=self._timeout) as ws:
            try:
                await ws.send_str(data)
            except Exception as e:
                raise APIConnectionError("failed to send request over WebSocket") from e

            while True:
                raw_msg = await ws.receive()
                if raw_msg.type == aiohttp.WSMsgType.ERROR:
                    exc = raw_msg.data
                    status_code = exc.status if isinstance(exc, aiohttp.ClientResponseError) else -1
                    raise APIStatusError(
                        str(exc), status_code=status_code, retryable=False
                    ) from exc
                if raw_msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "OpenAI Responses WebSocket connection closed unexpectedly",
                        status_code=raw_msg.data or -1,
                        body=f"{raw_msg.data=} {raw_msg.extra=}",
                    )
                if raw_msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(raw_msg.data)
                yield event
                if event["type"] in ["response.completed", "response.failed", "error"]:
                    return


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
    use_websocket: bool


class LLM(llm.LLM):
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
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        Create a new instance of OpenAI Responses LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """
        super().__init__()

        if not is_given(reasoning) and _supports_reasoning_effort(model):
            if model in ["gpt-5.1", "gpt-5.2", "gpt-5.4"]:
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
            use_websocket=use_websocket,
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

    async def aclose(self) -> None:
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
        )

    def _pending_tool_calls_completed(self, items: list[ChatItem]) -> bool:
        if not self._pending_tool_calls:
            return True

        completed_tool_calls = {
            item.call_id for item in items if item.type == "function_call_output"
        }
        return all(call_id in completed_tool_calls for call_id in self._pending_tool_calls)


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
        extra_kwargs: dict[str, Any],
        full_chat_ctx: llm.ChatContext,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._strict_tool_schema = strict_tool_schema
        self._response_id: str = ""
        self._response_completed: bool = False
        self._pending_tool_calls = set[str]()

        self._client = client
        self._llm: LLM = llm
        self._extra_kwargs = drop_unsupported_params(model, extra_kwargs)
        self._full_chat_ctx = full_chat_ctx.copy()

    async def _run(self) -> None:
        if self._llm._active_streams > 0:
            self._llm._parallel_generation = True
        self._llm._active_streams += 1
        try:
            await self._run_impl()
        finally:
            self._llm._active_streams -= 1
            if self._llm._active_streams == 0 and self._llm._parallel_generation:
                self._llm._prev_resp_id = ""
                self._llm._prev_chat_ctx = None
                self._llm._parallel_generation = False

    async def _run_impl(self) -> None:
        self._response_completed = False
        chat_ctx, _ = self._chat_ctx.to_provider_format(format="openai.responses")

        self._tool_ctx = llm.ToolContext(self.tools)
        tool_schemas = cast(
            list[ToolParam],
            self._tool_ctx.parse_function_tools(
                "openai.responses", strict=self._strict_tool_schema
            ),
        )

        if self._llm._opts.use_websocket is not False:
            retryable = True
            try:
                if self._llm._ws is None:
                    raise RuntimeError("use_websocket is True but _ws is None")

                payload = {
                    "type": "response.create",
                    "model": self._model,
                    "input": chat_ctx,
                    "tools": tool_schemas,
                    **self._extra_kwargs,
                }
                async for raw_event in self._llm._ws.generate_response(payload):
                    parsed_ev = self._parse_ws_event(raw_event)
                    self._process_event(parsed_ev)
                    retryable = False

                if not self._response_completed:
                    raise APIConnectionError(retryable=True)
            except (APIConnectionError, APIStatusError, APITimeoutError):
                raise
            except Exception as e:
                raise APIConnectionError(retryable=retryable) from e

        else:
            self._oai_stream: openai.AsyncStream[ResponseStreamEvent] | None = None
            retryable = True
            try:
                self._oai_stream = stream = cast(
                    openai.AsyncStream[ResponseStreamEvent],
                    await self._client.responses.create(  # type: ignore
                        model=self._model,
                        tools=tool_schemas,
                        input=cast(str | ResponseInputParam | openai.Omit, chat_ctx),
                        stream=True,
                        timeout=httpx.Timeout(self._conn_options.timeout),
                        **self._extra_kwargs,
                    ),
                )

                async with stream:
                    async for event in stream:
                        self._process_event(event)
                        retryable = False

            except openai.APITimeoutError:
                raise APITimeoutError(retryable=retryable)  # noqa: B904
            except openai.APIStatusError as e:
                raise APIStatusError(  # noqa: B904
                    e.message,
                    status_code=e.status_code,
                    request_id=e.request_id,
                    body=e.body,
                    retryable=retryable,
                )
            except (APIConnectionError, APIStatusError, APITimeoutError):
                raise
            except Exception as e:
                raise APIConnectionError(retryable=retryable) from e

    def _parse_ws_event(self, event: dict) -> ResponseStreamEvent | None:
        event_type = event.get("type", "")
        if event_type == "error":
            return ResponseErrorEvent.model_validate({**event.get("error", {}), **event})
        elif event_type == "response.created":
            return ResponseCreatedEvent.model_validate(event)
        elif event_type == "response.output_item.done":
            return ResponseOutputItemDoneEvent.model_validate(event)
        elif event_type == "response.output_text.delta":
            return ResponseTextDeltaEvent.model_validate(event)
        elif event_type == "response.completed":
            return ResponseCompletedEvent.model_validate(event)
        elif event_type == "response.failed":
            return ResponseFailedEvent.model_validate(event)
        return None

    def _process_event(self, event: ResponseStreamEvent | None) -> None:
        if event is None:
            return
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
        if chunk is not None:
            self._event_ch.send_nowait(chunk)

    def _handle_error(self, event: ResponseErrorEvent) -> None:
        error_code = -1
        try:
            error_code = int(event.code) if event.code else -1
        except ValueError:
            pass
        raise APIStatusError(event.message, status_code=error_code, retryable=False)

    def _handle_response_failed(self, event: ResponseFailedEvent) -> None:
        err = event.response.error
        raise APIStatusError(
            err.message if err else "response.failed",
            status_code=-1,
            retryable=False,
        )

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        self._response_id = event.response.id

    def _handle_response_completed(self, event: ResponseCompletedEvent) -> llm.ChatChunk | None:
        self._response_completed = True
        self._llm._prev_chat_ctx = self._full_chat_ctx
        self._llm._prev_resp_id = self._response_id
        self._llm._pending_tool_calls = self._pending_tool_calls

        chunk = None
        if usage := event.response.usage:
            chunk = llm.ChatChunk(
                id=self._response_id,
                usage=llm.CompletionUsage(
                    completion_tokens=usage.output_tokens,
                    prompt_tokens=usage.input_tokens,
                    prompt_cached_tokens=usage.input_tokens_details.cached_tokens
                    if usage.input_tokens_details
                    else 0,
                    total_tokens=usage.total_tokens,
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
        return chunk

    def _handle_response_output_text_delta(
        self, event: ResponseTextDeltaEvent
    ) -> llm.ChatChunk | None:
        return llm.ChatChunk(
            id=self._response_id,
            delta=llm.ChoiceDelta(content=event.delta, role="assistant"),
        )
