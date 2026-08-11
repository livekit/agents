from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from openai.types import Reasoning

from livekit.agents import APIConnectionError
from livekit.plugins.openai.responses.llm import (
    _WS_HEARTBEAT,
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

    async def send_str(self, data: str) -> None:
        if self._dead:
            raise ConnectionResetError("Cannot write to closing transport")
        self.sent = data

    async def receive(self) -> _FakeWSMsg:
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


class _StallingResponsesWS:
    """Replays raw response frames, then dies the way a socket going quiet does."""

    # read by LLM.provider
    _base_url = "https://api.openai.com/v1"

    def __init__(self, frames: list[dict]) -> None:
        self._frames = frames
        self.attempts = 0

    def generate_response(self, payload: dict):  # noqa: ANN201
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
