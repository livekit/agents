from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx2 as httpx
import pytest
from openai.types import Reasoning
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_stream_event import ResponseStreamEvent

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, llm as agents_llm
from livekit.agents.voice import generation
from livekit.agents.voice.io import ModelSettings
from livekit.plugins.openai.responses.llm import (
    _WS_HEARTBEAT,
    LLM as ResponsesLLM,
    LLMStream,
    _ResponsesWebsocket,
)

pytestmark = pytest.mark.plugin("openai")


class _FakeWSMsg:
    def __init__(self, data: str) -> None:
        self.type = aiohttp.WSMsgType.TEXT
        self.data = data
        self.extra = None


class _RecordingWS:
    """Minimal aiohttp-websocket stand-in: records what was sent, then replays
    a single terminal frame so `generate_response` returns. When ``dead`` is set,
    send_str fails the way a socket closed while idle does (and ws.closed stays
    False, mirroring aiohttp)."""

    def __init__(self, reply: dict, *, dead: bool = False) -> None:
        self.sent: str | None = None
        self._reply = reply
        self._dead = dead
        self.closed = False
        self.receive_calls = 0

    async def send_str(self, data: str) -> None:
        if self._dead:
            raise ConnectionResetError("Cannot write to closing transport")
        self.sent = data

    async def receive(self, timeout: float | None = None) -> _FakeWSMsg:
        self.receive_calls += 1
        if self.receive_calls > 1:
            raise AssertionError("terminal response must not read another WebSocket frame")
        return _FakeWSMsg(json.dumps(self._reply))

    async def close(self) -> None:
        self.closed = True


def _make_transport() -> _ResponsesWebsocket:
    return _ResponsesWebsocket(api_key="test-key", timeout=1.0, model="gpt-4.1")


def _use_connections(transport: _ResponsesWebsocket, *conns: _RecordingWS) -> None:
    """Wire a connect callback that hands out ``conns`` in order (last one repeats)."""
    remaining = list(conns)

    async def _connect(_timeout: float) -> _RecordingWS:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    transport._pool._connect_cb = _connect  # type: ignore[assignment]


async def _capture_sent_payload(msg: dict) -> dict:
    """Run the real `_ResponsesWebsocket.generate_response` against a real
    ConnectionPool whose connect callback yields a recording websocket, and
    return the JSON that was actually put on the wire."""
    transport = _make_transport()
    rec = _RecordingWS({"type": "response.completed", "response": {"output": []}})
    _use_connections(transport, rec)
    async for _ in transport.generate_response(msg):
        pass
    assert rec.sent is not None
    return json.loads(rec.sent)


async def test_reasoning_object_serialized_without_null_fields() -> None:
    """The WS transport serializes request models itself (not via the openai
    SDK). It must drop unset/None fields, otherwise Optional fields default to
    an explicit `null` on the wire and the Responses API 400s — e.g. after
    openai-python added `Reasoning.mode` (default None), `Reasoning(effort=...)`
    began emitting `"mode": null`, rejected with 'expected one of standard or
    pro, but got null instead.' Regression guard for that class of bug."""
    payload = {
        "type": "response.create",
        "model": "gpt-5.4",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "reasoning": Reasoning(effort="none"),
    }

    sent = await _capture_sent_payload(payload)

    assert sent["reasoning"] == {"effort": "none"}
    # No serialized request model may carry an explicit null-valued key.
    assert None not in sent["reasoning"].values()


async def test_incomplete_response_is_a_terminal_websocket_event() -> None:
    """`response.incomplete` ends a request just like completed/failed/error.

    When the transport generator is fully consumed directly, it must yield the
    frame once, return without waiting for a second frame, and make the cleanly
    terminated socket available for reuse.
    """
    frame = {"type": "response.incomplete"}
    rec = _RecordingWS(frame)
    transport = _make_transport()
    _use_connections(transport, rec)

    events = [event async for event in transport.generate_response({"type": "response.create"})]

    assert events == [frame]
    assert rec.receive_calls == 1
    assert rec in transport._pool._available
    await transport.aclose()


def test_error_event_missing_sequence_number_parses_cleanly() -> None:
    """Top-level protocol error frames (e.g. a request-validation 400) don't
    carry `sequence_number`, which ResponseErrorEvent marks required. Parsing
    must not raise a pydantic ValidationError that masks the real API message —
    it should surface the message so it reaches the caller as an APIStatusError."""
    frame = {
        "type": "error",
        "message": "Invalid type for 'reasoning.mode': expected one of "
        "'standard' or 'pro', but got null instead.",
        "code": "invalid_type",
        "param": "reasoning.mode",
        "status": 400,
    }

    # `_parse_ws_event` does not read `self`; invoke it directly on the frame.
    parsed = LLMStream._parse_ws_event(object(), frame)  # type: ignore[arg-type]

    assert parsed is not None
    assert parsed.type == "error"
    assert parsed.message == frame["message"]
    assert parsed.param == "reasoning.mode"


async def test_stale_reused_ws_is_discarded_and_request_succeeds() -> None:
    """Regression for #6513: a pooled WebSocket that OpenAI (or an intermediary
    such as a NAT/proxy) closed while idle only fails on send when reused —
    aiohttp keeps ws.closed False until a read observes the close. The transport
    must discard the stale connection and reconnect in place, instead of raising
    a retryable error that costs a full outer LLM retry (with backoff) for every
    stale socket."""
    reply = {"type": "response.completed", "response": {"output": []}}
    fresh = _RecordingWS(reply)

    transport = _make_transport()
    _use_connections(transport, fresh)

    # two pooled connections dropped while idle: not expired, send fails, and
    # ws.closed is still False (so a `not ws.closed` validation would not help).
    stale = [_RecordingWS(reply, dead=True), _RecordingWS(reply, dead=True)]
    for ws in stale:
        transport._pool._connections[ws] = time.time()  # type: ignore[index]
        transport._pool._available.add(ws)  # type: ignore[arg-type]

    async for _ in transport.generate_response({"type": "response.create"}):
        pass

    assert all(s.sent is None for s in stale), "stale sockets must not carry the request"
    assert fresh.sent is not None, "request must be sent on a fresh connection"
    await transport.aclose()


async def test_fresh_ws_send_failure_is_raised() -> None:
    """A brand-new connection that fails to send is a genuine error and must be
    surfaced, not silently retried the way a stale reused connection is."""
    reply = {"type": "response.completed", "response": {"output": []}}
    dead_fresh = _RecordingWS(reply, dead=True)

    transport = _make_transport()
    _use_connections(transport, dead_fresh)

    with pytest.raises(APIConnectionError):
        async for _ in transport.generate_response({"type": "response.create"}):
            pass
    await transport.aclose()


async def test_send_cancellation_discards_socket() -> None:
    """A cancellation during send must discard the acquired socket, not leave it
    orphaned in the pool (never reused, never closed until aclose)."""
    ws = _RecordingWS({"type": "response.completed", "response": {"output": []}})

    async def _cancel(_data: str) -> None:
        raise asyncio.CancelledError

    ws.send_str = _cancel  # type: ignore[method-assign]

    transport = _make_transport()
    _use_connections(transport, ws)

    with pytest.raises(asyncio.CancelledError):
        await transport._acquire_and_send("{}")

    assert ws not in transport._pool._connections
    assert ws not in transport._pool._available
    await transport.aclose()


async def test_create_ws_enables_heartbeat() -> None:
    """Pooled Responses sockets set a ws heartbeat so idle connections stay warm
    and dead peers are detected instead of surfacing only on the next send."""
    transport = _make_transport()
    fake_ws = object()
    session = MagicMock()
    session.ws_connect = AsyncMock(return_value=fake_ws)
    transport._ensure_http_session = lambda: session  # type: ignore[method-assign]

    assert await transport._create_ws(timeout=1.0) is fake_ws
    _, kwargs = session.ws_connect.call_args
    assert kwargs["heartbeat"] == _WS_HEARTBEAT


async def test_responses_llm_prewarms_websocket_pool() -> None:
    llm_model = ResponsesLLM(model="gpt-4.1", api_key="test-key")
    assert llm_model._ws is not None
    llm_model._ws._pool.prewarm = prewarm = MagicMock()  # type: ignore[method-assign]

    try:
        await llm_model._prewarm_impl()
        prewarm.assert_called_once_with()
    finally:
        await llm_model.aclose()


@pytest.mark.parametrize("connect_timeout", [0.05, 1.0])
async def test_responses_prewarm_uses_configured_connect_timeout(connect_timeout: float) -> None:
    llm_model = ResponsesLLM(
        model="gpt-4.1", api_key="test-key", timeout=httpx.Timeout(connect_timeout)
    )
    assert llm_model._ws is not None
    transport = llm_model._ws
    ws = _RecordingWS({"type": "response.completed"})
    timeouts: list[float] = []

    async def connect(timeout: float) -> _RecordingWS:
        timeouts.append(timeout)
        return ws

    transport._pool._connect_cb = connect  # type: ignore[assignment]
    try:
        await llm_model._prewarm_impl()
        assert transport._pool._prewarm_task is not None
        prewarm_task = transport._pool._prewarm_task()
        assert prewarm_task is not None
        await asyncio.wait_for(asyncio.shield(prewarm_task), timeout=1.0)
        assert timeouts == [connect_timeout]
        assert await transport._acquire_and_send("{}") is cast(aiohttp.ClientWebSocketResponse, ws)
        assert timeouts == [connect_timeout], "the first turn should reuse the prewarmed socket"
        transport._pool.put(ws)  # type: ignore[arg-type]
    finally:
        await llm_model.aclose()
    assert ws.closed


@pytest.mark.parametrize("cancel_request", [False, True])
async def test_responses_acquisition_preserves_shared_prewarm(
    cancel_request: bool,
) -> None:
    transport = _ResponsesWebsocket(api_key="test-key", timeout=0.05, model="gpt-4.1")
    started = asyncio.Event()
    release = asyncio.Event()
    ws = _RecordingWS({"type": "response.completed"})

    async def connect(_timeout: float) -> _RecordingWS:
        started.set()
        await release.wait()
        return ws

    transport._pool._connect_cb = connect  # type: ignore[assignment]
    transport._pool.prewarm()
    request: asyncio.Task[aiohttp.ClientWebSocketResponse] | None = None
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert transport._pool._prewarm_task is not None
        prewarm_task = transport._pool._prewarm_task()
        assert prewarm_task is not None
        request = asyncio.create_task(transport._acquire_and_send("{}"))
        if cancel_request:
            await asyncio.sleep(0)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        else:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(request), timeout=0.1)
            assert not request.done(), "acquisition should wait for the shared prewarm"

        assert not prewarm_task.done(), "a request must not cancel the shared prewarm"
        assert ws.sent is None
        release.set()
        await asyncio.wait_for(asyncio.shield(prewarm_task), timeout=1.0)
        if cancel_request:
            acquired = await transport._acquire_and_send("{}")
        else:
            acquired = await asyncio.wait_for(request, timeout=1.0)
        assert acquired is cast(aiohttp.ClientWebSocketResponse, ws)
        assert transport._pool.last_connection_reused
        transport._pool.put(ws)  # type: ignore[arg-type]
    finally:
        if request is not None and not request.done():
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        await transport.aclose()
    assert ws.closed


async def test_responses_llm_prewarms_http_client() -> None:
    client = MagicMock()
    client._base_url = httpx.URL("https://api.openai.com/v1")
    client.models.list = list_models = AsyncMock()
    llm_model = ResponsesLLM(model="gpt-4.1", client=client, use_websocket=False)

    try:
        await llm_model._prewarm_impl()
        list_models.assert_awaited_once_with()
    finally:
        await llm_model.aclose()


_RESPONSE_CREATED = {
    "type": "response.created",
    "sequence_number": 0,
    "response": {
        "id": "resp_1",
        "created_at": 0,
        "model": "gpt-4.1",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    },
}
_TEXT_DELTA = {
    "type": "response.output_text.delta",
    "sequence_number": 1,
    "content_index": 0,
    "delta": "hello",
    "item_id": "msg_1",
    "output_index": 0,
    "logprobs": [],
}


def _response_incomplete(reason: str | None) -> dict:
    response = {
        **_RESPONSE_CREATED["response"],
        "status": "incomplete",
    }
    if reason is not None:
        response["incomplete_details"] = {"reason": reason}
    return {
        "type": "response.incomplete",
        "sequence_number": 2,
        "response": response,
    }


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter"])
def test_incomplete_ws_event_parses_provider_reason(reason: str) -> None:
    parsed = LLMStream._parse_ws_event(  # type: ignore[arg-type]
        object(), _response_incomplete(reason)
    )

    assert isinstance(parsed, ResponseIncompleteEvent)
    assert parsed.response.incomplete_details is not None
    assert parsed.response.incomplete_details.reason == reason


class _ReplayResponsesWS:
    """Hermetic WebSocket LLM transport that ends after the supplied frames."""

    _base_url = "https://api.openai.com/v1"

    def __init__(self, frames: list[dict]) -> None:
        self._frames = frames
        self.attempts = 0

    def generate_response(  # noqa: ANN201
        self,
        payload: dict,
        *,
        receive_timeout: float | None = None,
        connect_timeout: float | None = None,
    ):
        self.attempts += 1

        async def _replay():  # noqa: ANN202
            for frame in self._frames:
                yield frame

        return _replay()

    async def aclose(self) -> None:
        pass


class _ReplayHTTPStream:
    """Minimal OpenAI AsyncStream stand-in for the HTTP Responses path."""

    def __init__(self, events: Sequence[ResponseStreamEvent]) -> None:
        self._events = events

    async def __aenter__(self) -> _ReplayHTTPStream:
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    def __aiter__(self):  # noqa: ANN204
        async def _replay():  # noqa: ANN202
            for event in self._events:
                yield event

        return _replay()


@pytest.mark.parametrize("transport_kind", ["websocket", "http"])
@pytest.mark.parametrize("reason", ["max_output_tokens", None])
async def test_incomplete_response_is_non_retryable_and_does_not_update_context(
    transport_kind: str, reason: str | None
) -> None:
    raw_events = [_RESPONSE_CREATED, _response_incomplete(reason)]

    if transport_kind == "websocket":
        llm_model = ResponsesLLM(model="gpt-4.1", api_key="test-key")
        transport = _ReplayResponsesWS(raw_events)
        llm_model._ws = transport  # type: ignore[assignment]
        request_counter = transport
    else:
        client = MagicMock()
        client._base_url = httpx.URL("https://api.openai.com/v1")
        parsed_events = [
            ResponseCreatedEvent.model_validate(raw_events[0]),
            ResponseIncompleteEvent.model_validate(raw_events[1]),
        ]
        client.responses.create = AsyncMock(
            side_effect=lambda **_kwargs: _ReplayHTTPStream(parsed_events)
        )
        llm_model = ResponsesLLM(model="gpt-4.1", client=client, use_websocket=False)
        request_counter = client.responses.create

    chat_ctx = agents_llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")

    try:
        with pytest.raises(APIStatusError) as exc_info:
            async with llm_model.chat(
                chat_ctx=chat_ctx,
                conn_options=APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=5.0),
            ) as stream:
                async for _ in stream:
                    pass

        error = exc_info.value
        assert error.status_code == -1
        assert error.retryable is False
        if reason is None:
            assert "reason unavailable" in error.message
        else:
            assert reason in error.message
        attempts = (
            request_counter.attempts
            if isinstance(request_counter, _ReplayResponsesWS)
            else request_counter.await_count
        )
        assert attempts == 1
        assert llm_model._prev_resp_id == ""
        assert llm_model._prev_resp_items is None
    finally:
        await llm_model.aclose()


class _StallingResponsesWS:
    """Replays raw response frames, then dies the way a socket going quiet does."""

    # read by LLM.provider
    _base_url = "https://api.openai.com/v1"

    def __init__(self, frames: list[dict]) -> None:
        self._frames = frames
        self.attempts = 0

    def generate_response(  # noqa: ANN201
        self,
        payload: dict,
        *,
        receive_timeout: float | None = None,
        connect_timeout: float | None = None,
    ):
        self.attempts += 1
        frames = self._frames

        async def _replay():  # noqa: ANN202
            for frame in frames:
                yield frame
            raise ConnectionResetError("stalled mid-stream")

        return _replay()

    async def aclose(self) -> None:
        pass


async def _attempts_until_stall(frames: list[dict], *, max_retry: int) -> int:
    from livekit.agents import APIConnectOptions, llm as agents_llm
    from livekit.plugins.openai.responses.llm import LLM as ResponsesLLM

    llm_model = ResponsesLLM(model="gpt-4.1", api_key="test-key")
    ws = _StallingResponsesWS(frames)
    llm_model._ws = ws  # type: ignore[assignment]

    chat_ctx = agents_llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")

    with pytest.raises(Exception):  # noqa: PT011, B017
        async with llm_model.chat(
            chat_ctx=chat_ctx,
            conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0, timeout=5.0),
        ) as stream:
            async for _ in stream:
                pass

    return ws.attempts


async def test_response_created_alone_stays_retryable() -> None:
    """`response.created` opens every stream and carries no output, so a stall
    right after it has surfaced nothing a retry could duplicate."""
    assert await _attempts_until_stall([_RESPONSE_CREATED], max_retry=2) == 3


async def test_generated_text_is_not_retried() -> None:
    """Text already delivered must not be regenerated."""
    assert await _attempts_until_stall([_RESPONSE_CREATED, _TEXT_DELTA], max_retry=2) == 1


def _response_completed_with_usage(**usage: dict) -> dict:
    return {
        "type": "response.completed",
        "sequence_number": 1,
        "response": {**_RESPONSE_CREATED["response"], "status": "completed", "usage": usage},
    }


class _UsageResponsesWS:
    """Records the chunk the stream sent for the frames it replays."""

    _base_url = "https://api.openai.com/v1"

    def __init__(self, frames: list[dict]) -> None:
        self._frames = frames
        self.receive_timeouts: list[float | None] = []

    def generate_response(  # noqa: ANN201
        self,
        payload: dict,
        *,
        receive_timeout: float | None = None,
        connect_timeout: float | None = None,
    ):
        self.receive_timeouts.append(receive_timeout)
        frames = self._frames

        async def _replay():  # noqa: ANN202
            for frame in frames:
                yield frame

        return _replay()

    async def aclose(self) -> None:
        pass


async def test_plugin_ws_receive_uses_conn_timeout() -> None:
    llm_model = ResponsesLLM(model="gpt-4.1", api_key="test-key")
    transport = _UsageResponsesWS(
        [
            _RESPONSE_CREATED,
            _response_completed_with_usage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
                output_tokens_details={"reasoning_tokens": 0},
            ),
        ]
    )
    llm_model._ws = transport  # type: ignore[assignment]
    chat_ctx = agents_llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")

    try:
        async with llm_model.chat(
            chat_ctx=chat_ctx, conn_options=APIConnectOptions(timeout=0.05)
        ) as stream:
            async for _ in stream:
                pass
    finally:
        await llm_model.aclose()

    assert transport.receive_timeouts == [0.05]


async def test_plugin_ws_receive_uses_stream_idle_timeout() -> None:
    llm_model = ResponsesLLM(model="gpt-4.1", api_key="test-key", stream_idle_timeout=2.0)
    transport = _UsageResponsesWS(
        [
            _RESPONSE_CREATED,
            _response_completed_with_usage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
                output_tokens_details={"reasoning_tokens": 0},
            ),
        ]
    )
    llm_model._ws = transport  # type: ignore[assignment]
    chat_ctx = agents_llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")

    try:
        async with llm_model.chat(
            chat_ctx=chat_ctx, conn_options=APIConnectOptions(timeout=0.05)
        ) as stream:
            async for _ in stream:
                pass
    finally:
        await llm_model.aclose()

    assert transport.receive_timeouts == [2.0]


def test_plugin_rejects_zero_stream_idle_timeout() -> None:
    with pytest.raises(ValueError, match="stream_idle_timeout"):
        ResponsesLLM(model="gpt-4.1", api_key="test-key", stream_idle_timeout=0)


async def _usage_from_frames(frames: list[dict]) -> agents_llm.CompletionUsage:
    llm_model = ResponsesLLM(model="gpt-4.1", api_key="test-key")
    llm_model._ws = _UsageResponsesWS(frames)  # type: ignore[assignment]

    chat_ctx = agents_llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")

    usage: agents_llm.CompletionUsage | None = None
    try:
        async with llm_model.chat(
            chat_ctx=chat_ctx, conn_options=APIConnectOptions(timeout=5.0)
        ) as stream:
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
    finally:
        await llm_model.aclose()

    assert usage is not None
    return usage


async def test_completed_response_reports_cache_write_tokens() -> None:
    """`response.usage.input_tokens_details.cache_write_tokens` is the Responses
    API's cache-write counterpart to the Chat Completions field of the same name.
    It is a subset of `input_tokens`, so it must not move any total."""
    usage = await _usage_from_frames(
        [
            _RESPONSE_CREATED,
            _response_completed_with_usage(
                input_tokens=2596,
                output_tokens=40,
                total_tokens=2636,
                input_tokens_details={"cached_tokens": 1024, "cache_write_tokens": 1536},
                output_tokens_details={"reasoning_tokens": 0},
            ),
        ]
    )

    assert usage.cache_creation_tokens == 1536
    assert usage.prompt_cached_tokens == 1024
    assert usage.prompt_tokens == 2596
    assert usage.total_tokens == 2636


async def test_completed_response_reports_reasoning_tokens() -> None:
    """`response.usage.output_tokens_details.reasoning_tokens` is the Responses API's
    counterpart to `completion_tokens_details.reasoning_tokens`. It is already part of
    `output_tokens`, so it must not move `completion_tokens` or `total_tokens`."""
    usage = await _usage_from_frames(
        [
            _RESPONSE_CREATED,
            _response_completed_with_usage(
                input_tokens=100,
                output_tokens=340,
                total_tokens=440,
                input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
                output_tokens_details={"reasoning_tokens": 300},
            ),
        ]
    )

    assert usage.reasoning_tokens == 300
    assert usage.completion_tokens == 340
    assert usage.total_tokens == 440


async def test_completed_response_without_cache_write_reports_zero() -> None:
    usage = await _usage_from_frames(
        [
            _RESPONSE_CREATED,
            _response_completed_with_usage(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
                output_tokens_details={"reasoning_tokens": 0},
            ),
        ]
    )

    assert usage.cache_creation_tokens == 0


class _ScriptedResponsesWS:
    """Answers each request with its own scripted response and records the payloads."""

    _base_url = "https://api.openai.com/v1"

    def __init__(self, outputs: list[list[dict]]) -> None:
        self._outputs = outputs
        self.payloads: list[dict] = []

    def generate_response(  # noqa: ANN201
        self,
        payload: dict,
        *,
        receive_timeout: float | None = None,
        connect_timeout: float | None = None,
    ):
        self.payloads.append(payload)
        resp_id = f"resp_{len(self.payloads)}"
        output = self._outputs[len(self.payloads) - 1]
        response = {**_RESPONSE_CREATED["response"], "id": resp_id, "status": "completed"}
        frames = [
            {"type": "response.created", "sequence_number": 0, "response": response},
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {**response, "output": output},
            },
        ]

        async def _replay():  # noqa: ANN202
            for frame in frames:
                yield frame

        return _replay()

    async def aclose(self) -> None:
        pass


def _output_message(text: str) -> dict:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _phase_message_events(
    item_id: str,
    text: str,
    output_index: int,
    added_fields: dict[str, str | None],
    done_fields: dict[str, str | None],
) -> tuple[dict, dict, dict]:
    message = {**_output_message(text), "id": item_id}
    return (
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {**message, "status": "in_progress", "content": [], **added_fields},
        },
        {**_TEXT_DELTA, "item_id": item_id, "output_index": output_index, "delta": text},
        {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {**message, **done_fields},
        },
    )


def _phase_stream_model(
    transport_kind: str, events: list[dict], early: bool | None = True
) -> ResponsesLLM:
    options = {} if early is None else {"emit_phase_on_item_start": early}
    completed = {
        "type": "response.completed",
        "response": {
            **_RESPONSE_CREATED["response"],
            "status": "completed",
            "output": [
                event["item"] for event in events if event["type"] == "response.output_item.done"
            ],
        },
    }
    frames = [
        {**event, "sequence_number": sequence}
        for sequence, event in enumerate([_RESPONSE_CREATED, *events, completed])
    ]
    if transport_kind == "websocket":
        llm_model = ResponsesLLM(api_key="test-key", **options)
        llm_model._ws = _ReplayResponsesWS(frames)  # type: ignore[assignment]
    else:
        client = MagicMock()
        client._base_url = httpx.URL("https://api.openai.com/v1")
        event_classes = {
            "response.created": ResponseCreatedEvent,
            "response.output_item.added": ResponseOutputItemAddedEvent,
            "response.output_text.delta": ResponseTextDeltaEvent,
            "response.output_item.done": ResponseOutputItemDoneEvent,
            "response.completed": ResponseCompletedEvent,
        }
        client.responses.create = AsyncMock(
            return_value=_ReplayHTTPStream(
                [event_classes[frame["type"]].model_validate(frame) for frame in frames]
            )
        )
        llm_model = ResponsesLLM(client=client, use_websocket=False, **options)

    return llm_model


async def _phase_stream_deltas(
    transport_kind: str, events: list[dict], early: bool | None = True
) -> list[agents_llm.ChoiceDelta]:
    llm_model = _phase_stream_model(transport_kind, events, early)
    chat_ctx = agents_llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")
    try:
        async with llm_model.chat(
            chat_ctx=chat_ctx, conn_options=APIConnectOptions(max_retry=0)
        ) as stream:
            return [chunk.delta async for chunk in stream if chunk.delta is not None]
    finally:
        await llm_model.aclose()


def _phase_delta(
    item_id: str, phase: str | None = None, *, content: str | None = None
) -> agents_llm.ChoiceDelta:
    metadata = {"item_id": item_id}
    if phase is not None:
        metadata["phase"] = phase
    extra = {"openai_item": metadata}
    if content is None and phase is not None:
        extra["openai"] = {"phase": phase}
    return agents_llm.ChoiceDelta(role="assistant", content=content, extra=extra)


@pytest.mark.parametrize("transport_kind", ["websocket", "http"])
@pytest.mark.parametrize("early", [None, False])
async def test_output_message_phase_defaults_to_completion(
    transport_kind: str, early: bool | None
) -> None:
    events = _phase_message_events(
        "msg_1", "hello", 0, {"phase": "commentary"}, {"phase": "final_answer"}
    )

    assert await _phase_stream_deltas(transport_kind, list(events), early) == [
        agents_llm.ChoiceDelta(role="assistant", content="hello"),
        agents_llm.ChoiceDelta(role="assistant", extra={"openai": {"phase": "final_answer"}}),
    ]


@pytest.mark.parametrize("transport_kind", ["websocket", "http"])
@pytest.mark.parametrize(
    "added_fields, done_fields, expected",
    [
        pytest.param(
            {"phase": "commentary"},
            {"phase": "commentary"},
            [
                _phase_delta("msg_1", "commentary"),
                _phase_delta("msg_1", "commentary", content="hello"),
            ],
            id="same-phase",
        ),
        pytest.param(
            {"phase": "final_answer"},
            {},
            [
                _phase_delta("msg_1", "final_answer"),
                _phase_delta("msg_1", "final_answer", content="hello"),
            ],
            id="phase-only-at-start",
        ),
        pytest.param(
            {"phase": None},
            {"phase": "final_answer"},
            [_phase_delta("msg_1", content="hello"), _phase_delta("msg_1", "final_answer")],
            id="late-phase",
        ),
        pytest.param(
            {"phase": "commentary"},
            {"phase": "final_answer"},
            [
                _phase_delta("msg_1", "commentary"),
                _phase_delta("msg_1", "commentary", content="hello"),
                _phase_delta("msg_1", "final_answer"),
            ],
            id="changed-phase",
        ),
        pytest.param({}, {"phase": None}, [_phase_delta("msg_1", content="hello")], id="no-phase"),
    ],
)
async def test_output_message_phase_order(
    transport_kind: str,
    added_fields: dict[str, str | None],
    done_fields: dict[str, str | None],
    expected: list[agents_llm.ChoiceDelta],
) -> None:
    events = _phase_message_events("msg_1", "hello", 0, added_fields, done_fields)
    assert await _phase_stream_deltas(transport_kind, list(events)) == expected


@pytest.mark.parametrize("transport_kind", ["websocket", "http"])
async def test_output_message_phase_is_scoped_to_its_item(transport_kind: str) -> None:
    commentary = {"phase": "commentary"}
    add_a, text_a, done_a = _phase_message_events("msg_a", "working", 0, commentary, commentary)
    add_b, text_b, done_b = _phase_message_events("msg_b", "unknown", 1, {}, {})
    add_c, text_c, done_c = _phase_message_events("msg_c", "more", 2, commentary, commentary)

    assert await _phase_stream_deltas(
        transport_kind, [add_a, add_a, add_b, text_b, text_a, done_b, done_a, add_c, text_c, done_c]
    ) == [
        _phase_delta("msg_a", "commentary"),
        _phase_delta("msg_b", content="unknown"),
        _phase_delta("msg_a", "commentary", content="working"),
        _phase_delta("msg_c", "commentary"),
        _phase_delta("msg_c", "commentary", content="more"),
    ]


@pytest.mark.parametrize("transport_kind", ["websocket", "http"])
async def test_output_message_phase_resets_on_response_created(transport_kind: str) -> None:
    add_old, _, _ = _phase_message_events("msg_1", "old", 0, {"phase": "commentary"}, {})
    add_new, text_new, done_new = _phase_message_events(
        "msg_1", "new", 0, {}, {"phase": "commentary"}
    )
    created = {
        **_RESPONSE_CREATED,
        "response": {**_RESPONSE_CREATED["response"], "id": "resp_next"},
    }

    assert await _phase_stream_deltas(
        transport_kind, [add_old, created, add_new, text_new, done_new]
    ) == [
        _phase_delta("msg_1", "commentary"),
        _phase_delta("msg_1", content="new"),
        _phase_delta("msg_1", "commentary"),
    ]


@pytest.mark.parametrize("transport_kind", ["websocket", "http"])
@pytest.mark.parametrize("consumer", ["collect", "voice"])
async def test_unphased_output_preserves_aggregated_phase(
    transport_kind: str, consumer: str
) -> None:
    phase = {"phase": "commentary"}
    phased = _phase_message_events("msg_a", "Working", 0, phase, phase)
    unphased = _phase_message_events("msg_b", " more", 1, {}, {})
    llm_model = _phase_stream_model(transport_kind, [*phased, *unphased])
    chat_ctx = agents_llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")
    stream = llm_model.chat(chat_ctx=chat_ctx, conn_options=APIConnectOptions(max_retry=0))

    async def node(
        _chat_ctx: agents_llm.ChatContext, _tools: list[agents_llm.Tool], _settings: ModelSettings
    ) -> AsyncIterator[agents_llm.ChatChunk]:
        async with stream:
            async for chunk in stream:
                yield chunk

    try:
        if consumer == "collect":
            response = await stream.collect()
            text, extra = response.text, response.extra
        else:
            task, data = generation.perform_llm_inference(
                node=node,
                chat_ctx=chat_ctx,
                tool_ctx=agents_llm.ToolContext([]),
                model_settings=ModelSettings(),
            )
            assert await task
            text, extra = data.generated_text, data.generated_extra
    finally:
        await llm_model.aclose()

    assert text == "Working more"
    assert extra.get("openai") == phase
    chat_ctx.add_message(role="assistant", content=text, extra=extra)
    items, _ = chat_ctx.to_provider_format(format="openai.responses")
    assert items[-1]["phase"] == "commentary"


def _output_function_call(call_id: str, name: str) -> dict:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": '{"city": "Paris"}',
        "status": "completed",
    }


async def test_chained_request_sends_only_items_after_previous_response() -> None:
    ws = _ScriptedResponsesWS(
        [
            [
                _output_function_call("call_1", "get_weather"),
                _output_message("Let me check."),
                _output_function_call("call_2", "get_time"),
            ],
            [_output_message("It's sunny and 3pm, and tomorrow looks fine.")],
            [_output_message("You're welcome.")],
            [],
        ]
    )
    llm_model = ResponsesLLM(model="gpt-4.1", api_key="test-key")
    llm_model._ws = ws  # type: ignore[assignment]

    async def _turn() -> None:
        async with llm_model.chat(
            chat_ctx=chat_ctx, conn_options=APIConnectOptions(max_retry=0, timeout=5.0)
        ) as stream:
            async for _ in stream:
                pass

    chat_ctx = agents_llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="Weather and time in Paris?")
    await _turn()

    # stored the way the agent stores a generation: one message, calls sharing its id, and
    # each output right after its call
    chat_ctx.add_message(role="assistant", content="Let me check.", id="item_a")
    for i, (call_id, name, output) in enumerate(
        [("call_1", "get_weather", "sunny"), ("call_2", "get_time", "3pm")]
    ):
        chat_ctx.items.append(
            agents_llm.FunctionCall(
                id=f"item_a/fnc_{i}", call_id=call_id, name=name, arguments='{"city": "Paris"}'
            )
        )
        chat_ctx.items.append(
            agents_llm.FunctionCallOutput(call_id=call_id, name=name, output=output, is_error=False)
        )
    await _turn()

    # an interrupted reply differs from what the server holds
    chat_ctx.add_message(role="assistant", content="It's sunny and 3pm,", interrupted=True)
    chat_ctx.add_message(role="user", content="Thanks!")
    await _turn()

    chat_ctx.add_message(role="assistant", content="You're welcome.")
    chat_ctx.add_message(role="user", content="Bye")
    await _turn()
    await llm_model.aclose()

    first, tools, interrupted, last = ws.payloads
    assert "previous_response_id" not in first
    assert tools["previous_response_id"] == "resp_1"
    assert tools["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        {"type": "function_call_output", "call_id": "call_2", "output": "3pm"},
    ]
    assert "previous_response_id" not in interrupted
    assert len(interrupted["input"]) == 8
    assert last["previous_response_id"] == "resp_3"
    assert last["input"] == [{"role": "user", "content": "Bye"}]
