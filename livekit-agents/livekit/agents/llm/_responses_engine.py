from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncGenerator
from typing import Any, Protocol, cast

import aiohttp
import httpx2 as httpx
import openai
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
    ToolParam,
)
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.shared_params import ResponsesModel
from yarl import URL

from .. import llm, utils
from .._exceptions import APIConnectionError, APIStatusError, APITimeoutError
from ..inference.llm import drop_unsupported_params
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from .chat_context import ChatContext
from .tool_context import Tool

_OPENAI_HOST = "api.openai.com"
OPENAI_RESPONSES_WS_URL = f"wss://{_OPENAI_HOST}/v1/responses"

# ws ping interval; keeps idle pooled sockets warm and lets aiohttp detect dead peers
_WS_HEARTBEAT = 30.0
# max connections to try when a reused socket is stale, before the outer retry takes over
_WS_SEND_MAX_ATTEMPTS = 6


def check_idle_timeout(timeout: float | None) -> float | None:
    if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
        raise ValueError("stream_idle_timeout must be a finite number greater than 0")
    return timeout


class ResponsesWebsocket:
    def __init__(
        self,
        api_key: str | None,
        timeout: float | None,
        model: str,
        base_url: str | None = None,
        max_session_duration: float = 3600,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout or DEFAULT_API_CONNECT_OPTIONS.timeout
        url = URL(base_url if base_url else OPENAI_RESPONSES_WS_URL)
        if url.scheme in ("http", "https"):
            url = url.with_scheme("ws" if url.scheme == "http" else "wss")
        if url.host != _OPENAI_HOST:
            # OpenAI's native endpoint takes the model in the response.create
            # payload; gateways need it on the upgrade URL to route the
            # connection before the first frame.
            url = url.update_query(model=model)
        self._base_url = str(url)

        self._session: aiohttp.ClientSession | None = None

        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._create_ws,
            close_cb=self._close_ws,
            max_session_duration=max_session_duration,
            connect_timeout=self._timeout,
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
                    heartbeat=_WS_HEARTBEAT,
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

    async def generate_response(
        self,
        msg: dict,
        *,
        receive_timeout: float | None = None,
        connect_timeout: float | None = None,
    ) -> AsyncGenerator[dict, None]:
        def _default(o: object) -> object:
            if isinstance(o, openai.BaseModel):
                # exclude_none is load-bearing, not cosmetic. This hand-rolled WS
                # transport serializes request models itself instead of going
                # through the openai SDK (which omits unset fields). Without
                # exclude_none, every Optional field the model defaults to None
                # is emitted as an explicit `null` on the wire. The Responses API
                # rejects explicit nulls on fields that expect an enum: e.g. after
                # openai-python added `Reasoning.mode` (default None), a plain
                # `Reasoning(effort=...)` began serializing `"mode": null`, which
                # the API 400s with "Invalid type for 'reasoning.mode': expected
                # one of 'standard' or 'pro', but got null instead." Omitting None
                # mirrors the SDK's on-the-wire shape and is forward-compatible
                # with future Optional additions to these models.
                return o.model_dump(mode="json", exclude_none=True)
            raise TypeError(f"unexpected type {type(o)}")

        try:
            data = json.dumps(msg, default=_default)
        except TypeError as e:
            raise APIConnectionError(f"failed to serialize request: {e}") from e

        ws = await self._acquire_and_send(data, connect_timeout)
        completed = False
        loop = asyncio.get_running_loop()
        deadline = _next_deadline(loop, receive_timeout)
        try:
            while True:
                raw_msg = await _receive_before(loop, ws, deadline)
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
                # restart once the consumer asks for more, so its own time doesn't count
                deadline = _next_deadline(loop, receive_timeout)
                if event["type"] in [
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                    "error",
                ]:
                    completed = True
                    return
        finally:
            # only a cleanly completed exchange is safe to reuse; discard on any error
            if completed:
                self._pool.put(ws)
            else:
                self._pool.remove(ws)

    async def _acquire_and_send(
        self, data: str, connect_timeout: float | None = None
    ) -> aiohttp.ClientWebSocketResponse:
        # a socket closed while idle surfaces only as a send failure on reuse
        last_exc: Exception | None = None
        for _ in range(_WS_SEND_MAX_ATTEMPTS):
            ws = await self._pool.get(
                timeout=self._timeout if connect_timeout is None else connect_timeout
            )
            reused = self._pool.last_connection_reused
            try:
                await ws.send_str(data)
                return ws
            except Exception as e:
                self._pool.remove(ws)  # discard the failed socket
                last_exc = e
                if not reused:
                    break  # a fresh connection failing to send is a real error, not staleness
            except BaseException:
                self._pool.remove(ws)  # cancellation: discard the socket, don't leak it
                raise
        raise APIConnectionError("failed to send request over WebSocket") from last_exc


def _next_deadline(loop: asyncio.AbstractEventLoop, timeout: float | None) -> float | None:
    return None if timeout is None else loop.time() + timeout


async def _receive_before(
    loop: asyncio.AbstractEventLoop,
    ws: aiohttp.ClientWebSocketResponse,
    deadline: float | None,
) -> aiohttp.WSMessage:
    # receive(timeout=...) restarts its timer on every frame, including the heartbeat
    # pongs it consumes internally, so it can't bound the silence between events
    if deadline is None:
        return await ws.receive()
    try:
        return await asyncio.wait_for(ws.receive(), max(deadline - loop.time(), 0))
    except asyncio.TimeoutError as e:
        raise APITimeoutError("timed out waiting for a Responses WebSocket event") from e


class _ResponsesOptions(Protocol):
    @property
    def use_websocket(self) -> bool: ...


class ResponsesLLMHost(Protocol):
    """State that ResponsesLLMStream reads from the LLM that created it."""

    _opts: _ResponsesOptions
    _ws: ResponsesWebsocket | None
    _provider_tool_type: type[llm.ProviderTool]
    _active_streams: int
    _parallel_generation: bool
    _prev_resp_id: str
    _prev_chat_ctx: ChatContext | None
    _pending_tool_calls: set[str]
    _stream_idle_timeout: float | None


class ResponsesLLMStream(llm.LLMStream):
    _llm: ResponsesLLMHost  # type: ignore[assignment]

    def __init__(
        self,
        llm: llm.LLM,
        *,
        model: str | ResponsesModel,
        strict_tool_schema: bool,
        client: openai.AsyncClient | None,
        chat_ctx: llm.ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
        full_chat_ctx: llm.ChatContext,
        prompt_cache_breakpoints: bool,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._prompt_cache_breakpoints = prompt_cache_breakpoints
        self._model = model
        self._strict_tool_schema = strict_tool_schema
        self._response_id: str = ""
        self._response_completed: bool = False
        self._pending_tool_calls = set[str]()

        self._client = client
        self._llm = cast(ResponsesLLMHost, llm)
        # bound at construction: a gateway transport routes by the model on its URL, so a
        # turn must keep the transport matching the model it sends, across retries
        self._ws = self._llm._ws
        self._extra_kwargs = drop_unsupported_params(model, extra_kwargs)
        self._full_chat_ctx = full_chat_ctx.copy()

    def _idle_timeout(self) -> float:
        # only text frames reset this; transport pings and pongs don't
        if self._llm._stream_idle_timeout is not None:
            return self._llm._stream_idle_timeout
        return self._conn_options.timeout

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
        # Azure's HTTP endpoint can return several alternate `message` items in one
        # response; only the first that emits text is spoken.
        self._text_owner: str | None = None
        self._message_phases: dict[str, str | None] = {}
        self._second_message_started = False
        self._dropped_item_ids: list[str] = []
        self._hosted_continuation = False
        chat_ctx, _ = self._chat_ctx.to_provider_format(
            format="openai.responses", prompt_cache_breakpoints=self._prompt_cache_breakpoints
        )
        self._tool_ctx = llm.ToolContext(self.tools)
        tool_schemas = cast(
            list[ToolParam],
            self._tool_ctx.parse_function_tools(
                "openai.responses",
                strict=self._strict_tool_schema,
                provider_tool_type=self._llm._provider_tool_type,
            ),
        )

        if self._llm._opts.use_websocket is not False:
            retryable = True
            try:
                if self._ws is None:
                    raise RuntimeError("use_websocket is True but _ws is None")

                payload = {
                    "type": "response.create",
                    "model": self._model,
                    "input": chat_ctx,
                    "tools": tool_schemas,
                    **self._extra_kwargs,
                }
                async for raw_event in self._ws.generate_response(
                    payload,
                    receive_timeout=self._idle_timeout(),
                    connect_timeout=self._conn_options.timeout,
                ):
                    parsed_ev = self._parse_ws_event(raw_event)
                    chunk = self._process_event(parsed_ev)
                    if chunk is not None and chunk.has_response():
                        retryable = False

                if not self._response_completed:
                    raise APIConnectionError(retryable=True)
            except (APIConnectionError, APIStatusError, APITimeoutError) as e:
                if not retryable:
                    e.retryable = False  # text already reached the caller; a retry repeats it
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
                        timeout=httpx.Timeout(
                            self._conn_options.timeout, read=self._idle_timeout()
                        ),
                        **self._extra_kwargs,
                    ),
                )

                async with stream:
                    async for event in stream:
                        chunk = self._process_event(event)
                        if chunk is not None and chunk.has_response():
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
            except (APIConnectionError, APIStatusError, APITimeoutError) as e:
                if not retryable:
                    e.retryable = False  # text already reached the caller; a retry repeats it
                raise
            except Exception as e:
                raise APIConnectionError(retryable=retryable) from e

    def _parse_ws_event(self, event: dict) -> ResponseStreamEvent | None:
        # Strip prompt_cache_retention from any response object before validation:
        # the OpenAI SDK Pydantic type doesn't match actual API values (e.g. "in_memory"
        # vs "in-memory"). We don't use this field so dropping it is safe.
        if (
            isinstance(event.get("response"), dict)
            and "prompt_cache_retention" in event["response"]
        ):
            event = {
                **event,
                "response": {
                    k: v for k, v in event["response"].items() if k != "prompt_cache_retention"
                },
            }

        event_type = event.get("type", "")
        if event_type == "error":
            # Top-level protocol error frames (e.g. a request-validation 400) do
            # NOT carry `sequence_number`, which ResponseErrorEvent marks required.
            # Validating them as-is raises a pydantic ValidationError that masks
            # the real API message ("... 1 validation error ... sequence_number
            # Field required ..."). Default the field so the genuine error
            # surfaces as a clean APIStatusError via _handle_error instead.
            merged = {"sequence_number": -1, **event.get("error", {}), **event}
            return ResponseErrorEvent.model_validate(merged)
        elif event_type == "response.created":
            return ResponseCreatedEvent.model_validate(event)
        elif event_type == "response.output_item.added":
            # in-progress tool items can lack fields the done event carries
            # (web_search_call's `action`), so validate only the messages the stream reads
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") != "message":
                return None
            return ResponseOutputItemAddedEvent.model_validate(event)
        elif event_type == "response.output_item.done":
            return ResponseOutputItemDoneEvent.model_validate(event)
        elif event_type == "response.output_text.delta":
            return ResponseTextDeltaEvent.model_validate(event)
        elif event_type == "response.completed":
            return ResponseCompletedEvent.model_validate(event)
        elif event_type == "response.failed":
            return ResponseFailedEvent.model_validate(event)
        elif event_type == "response.incomplete":
            return ResponseIncompleteEvent.model_validate(event)
        return None

    def _process_event(self, event: ResponseStreamEvent | None) -> llm.ChatChunk | None:
        """Handle one stream event, returning the chunk it sent to the caller, if any."""
        if event is None:
            return None
        chunk = None
        if isinstance(event, ResponseErrorEvent):
            self._handle_error(event)
        if isinstance(event, ResponseCreatedEvent):
            self._handle_response_created(event)
        if isinstance(event, ResponseOutputItemAddedEvent):
            self._handle_output_item_added(event)
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
            self._event_ch.send_nowait(chunk)
        return chunk

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

    def _handle_response_incomplete(self, event: ResponseIncompleteEvent) -> None:
        details = event.response.incomplete_details
        reason = details.reason if details else None
        raise APIStatusError(
            f"response incomplete: {reason or 'reason unavailable'}",
            status_code=-1,
            retryable=False,
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

        if self._dropped_item_ids:
            logger.warning(
                "dropped extra Responses output items",
                extra={
                    "response_id": self._response_id,
                    "model": self._model,
                    "dropped_item_ids": self._dropped_item_ids,
                },
            )

        self._response_completed = True
        self._llm._prev_chat_ctx = self._full_chat_ctx
        # the stored response still holds the dropped messages; chaining onto it would
        # make the model believe it said them
        self._llm._prev_resp_id = "" if self._dropped_item_ids else self._response_id
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
                    cache_creation_tokens=usage.input_tokens_details.cache_write_tokens
                    if usage.input_tokens_details
                    else 0,
                    reasoning_tokens=usage.output_tokens_details.reasoning_tokens
                    if usage.output_tokens_details
                    else 0,
                    total_tokens=usage.total_tokens,
                    service_tier=getattr(event.response, "service_tier", None),
                ),
            )
        return chunk

    def _handle_output_item_added(self, event: ResponseOutputItemAddedEvent) -> None:
        """Record a message item's phase; whether it is spoken is decided at its first text."""
        if isinstance(event.item, ResponseOutputMessage):
            self._message_phases[event.item.id] = event.item.phase

    def _drop_item(self, item_id: str) -> None:
        """Mark an output item as an alternate answer that is not sent to the caller."""
        self._second_message_started = True
        if item_id not in self._dropped_item_ids:
            self._dropped_item_ids.append(item_id)

    def _handle_output_items_done(self, event: ResponseOutputItemDoneEvent) -> llm.ChatChunk | None:
        chunk = None
        if (
            self._text_owner is not None
            and not self._second_message_started
            and event.item.type not in ("message", "function_call", "reasoning")
        ):
            # the reply continues after server-side tools, so the next message is
            # the answer rather than an alternate; several calls in a row still
            # admit only that one message
            self._hosted_continuation = True
        if event.item.type == "function_call":
            # kept even after an alternate message was dropped: losing a call the
            # model made is worse than acting on a reply branch that wasn't spoken
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
            self._message_phases[event.item.id] = event.item.phase
            if event.item.id in self._dropped_item_ids:
                return None
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
        if not event.delta:
            return None
        text = event.delta
        if self._text_owner is None:
            self._text_owner = event.item_id
        elif event.item_id != self._text_owner:
            # a message counts as a second answer only once it emits text; the
            # owner's phase may arrive on its output_item.done, so read it here
            if event.item_id in self._dropped_item_ids:
                return None
            if self._hosted_continuation:
                self._hosted_continuation = False
            elif self._message_phases.get(self._text_owner) != "commentary":
                self._drop_item(event.item_id)
                return None
            # commentary or a hosted tool call and the answer after it are spoken as
            # one reply; the space keeps their sentences from running together
            self._text_owner = event.item_id
            text = " " + text
        return llm.ChatChunk(
            id=self._response_id,
            delta=llm.ChoiceDelta(content=text, role="assistant"),
        )
