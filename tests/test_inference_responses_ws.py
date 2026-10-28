from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp
import pytest
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from livekit.agents import APIConnectOptions, APIStatusError, APITimeoutError, llm as agents_llm
from livekit.agents.inference import responses as inference_responses
from livekit.agents.inference.responses import LLM as ResponsesLLM
from livekit.agents.llm._responses_engine import ResponsesWebsocket

pytestmark = pytest.mark.unit

_RESPONSE = {
    "id": "resp_1",
    "created_at": 0,
    "model": "gpt-4.1",
    "object": "response",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}
_USAGE = {
    "input_tokens": 120,
    "output_tokens": 30,
    "total_tokens": 150,
    "input_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}
_CONNECTED = {
    "type": "gateway.connected",
    "deployment": "standard_openai",
    "model": "gpt-4.1",
    "session_id": "sess_1",
}
_TEXT_FRAMES = [
    {"type": "response.created", "sequence_number": 0, "response": _RESPONSE},
    {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "output_index": 0,
        "content_index": 0,
        "item_id": "msg_1",
        "delta": "hi",
        "logprobs": [],
    },
    {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {**_RESPONSE, "status": "completed", "usage": _USAGE},
    },
]
_INVALID_EVENT = {
    "type": "gateway.error",
    "code": "invalid_event",
    "message": "unsupported event type",
}


class _FakeWsMsg:
    def __init__(self, data: str) -> None:
        self.type = aiohttp.WSMsgType.TEXT
        self.data = data
        self.extra = None


class _ScriptedWs:
    def __init__(self, frames: list[dict]) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._frames = list(frames)

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self, timeout: float | None = None) -> _FakeWsMsg:
        return _FakeWsMsg(json.dumps(self._frames.pop(0)))

    async def close(self) -> None:
        self.closed = True


class _SilentWs(_ScriptedWs):
    """Never answers."""

    async def receive(self, timeout: float | None = None) -> _FakeWsMsg:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _PongingWs(_ScriptedWs):
    """Never returns a frame, but keeps the socket busy the way heartbeat pongs do."""

    def __init__(self) -> None:
        super().__init__([])
        self.pongs = 0

    async def receive(self, timeout: float | None = None) -> _FakeWsMsg:
        # aiohttp consumes pongs inside receive() and restarts its own timer on each
        while True:
            await asyncio.sleep(0.01)
            self.pongs += 1


class _DialRecord:
    def __init__(self, url: str, headers: dict[str, str], heartbeat: float) -> None:
        self.url = url
        self.headers = headers
        self.heartbeat = heartbeat


class _FakeSession:
    def __init__(self) -> None:
        self.dials: list[_DialRecord] = []

    async def ws_connect(
        self, url: str, *, headers: dict[str, str], heartbeat: float
    ) -> _ScriptedWs:
        self.dials.append(_DialRecord(url, headers, heartbeat))
        return _ScriptedWs([])


class _RefusingSession:
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self._status = status
        self._headers = headers or {}

    async def ws_connect(
        self, url: str, *, headers: dict[str, str], heartbeat: float
    ) -> _ScriptedWs:
        request_url = URL(url)
        request_info = aiohttp.RequestInfo(
            url=request_url,
            method="GET",
            headers=CIMultiDictProxy(CIMultiDict()),
            real_url=request_url,
        )
        raise aiohttp.WSServerHandshakeError(
            request_info,
            (),
            status=self._status,
            message="Invalid response status",
            headers=CIMultiDict(self._headers),
        )


def _make_llm(**kwargs: Any) -> ResponsesLLM:
    return ResponsesLLM(
        "openai/gpt-4.1",
        base_url="https://gw.test/v1",
        api_key="k",
        api_secret="s",
        use_websocket=True,
        **kwargs,
    )


def _transport(llm_model: ResponsesLLM) -> inference_responses._GatewayResponsesWebsocket:
    transport = llm_model._ws
    assert isinstance(transport, inference_responses._GatewayResponsesWebsocket)
    return transport


def _dial_session(llm_model: ResponsesLLM) -> _FakeSession:
    session = _FakeSession()
    _transport(llm_model)._session = session  # type: ignore[assignment]
    return session


def _use_connection(llm_model: ResponsesLLM, ws: _ScriptedWs) -> None:
    async def _connect(_timeout: float) -> _ScriptedWs:
        return ws

    _transport(llm_model)._pool._connect_cb = _connect  # type: ignore[assignment]


def _ctx() -> agents_llm.ChatContext:
    chat_ctx = agents_llm.ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    return chat_ctx


async def _consume(stream: agents_llm.LLMStream) -> None:
    async for _ in stream:
        pass


async def _drain(llm_model: ResponsesLLM) -> str:
    stream = llm_model.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0))
    try:
        chunks = [chunk async for chunk in stream]
    finally:
        await stream.aclose()
    return "".join(c.delta.content or "" for c in chunks if c.delta)


async def test_ws_url_is_gateway_responses_with_model() -> None:
    llm_model = _make_llm()
    session = _dial_session(llm_model)

    await _transport(llm_model)._create_ws(1.0)

    expected = str(URL("wss://gw.test/v1/responses").update_query(model="openai/gpt-4.1"))
    assert session.dials[0].url == expected
    assert session.dials[0].heartbeat == 30.0


async def test_ws_dial_sends_fresh_token_and_inference_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def _counting_token(api_key: str | None, api_secret: str | None, ttl: float = 600) -> str:
        nonlocal calls
        calls += 1
        return f"token-{calls}"

    monkeypatch.setattr(inference_responses, "create_access_token", _counting_token)
    llm_model = _make_llm()
    session = _dial_session(llm_model)

    await _transport(llm_model)._create_ws(1.0)
    await _transport(llm_model)._create_ws(1.0)

    assert [d.headers["Authorization"] for d in session.dials] == [
        "Bearer token-1",
        "Bearer token-2",
    ]
    assert session.dials[0].headers["User-Agent"].startswith("LiveKit Agents/")


async def test_ws_dial_sends_provider_header() -> None:
    llm_model = _make_llm(provider="azure")
    session = _dial_session(llm_model)

    await _transport(llm_model)._create_ws(1.0)

    assert session.dials[0].headers["X-LiveKit-Inference-Provider"] == "azure"


async def test_ws_dial_sends_inference_class_header() -> None:
    llm_model = _make_llm(inference_class="priority")
    session = _dial_session(llm_model)

    await _transport(llm_model)._create_ws(1.0)

    assert session.dials[0].headers["X-LiveKit-Inference-Priority"] == "priority"


async def test_ws_per_call_inference_class_warns(caplog: pytest.LogCaptureFixture) -> None:
    llm_model = _make_llm()
    session = _dial_session(llm_model)

    with caplog.at_level(logging.WARNING):
        stream = llm_model.chat(chat_ctx=_ctx(), inference_class="priority")
    await stream.aclose()
    await _transport(llm_model)._create_ws(1.0)

    records = [r for r in caplog.records if "cannot change per request" in r.getMessage()]
    assert len(records) == 1
    assert records[0].requested == "priority"  # type: ignore[attr-defined]
    assert records[0].configured is None  # type: ignore[attr-defined]
    assert "X-LiveKit-Inference-Priority" not in session.dials[0].headers
    await llm_model.aclose()


async def test_ws_keepalive_mid_stream_is_ignored() -> None:
    llm_model = _make_llm()
    created, *rest = _TEXT_FRAMES
    _use_connection(
        llm_model, _ScriptedWs([_CONNECTED, created, {"type": "gateway.keepalive"}, *rest])
    )

    assert await _drain(llm_model) == "hi"
    await llm_model.aclose()


async def test_ws_receive_uses_stream_idle_timeout() -> None:
    llm_model = _make_llm(stream_idle_timeout=0.05)
    _use_connection(llm_model, _SilentWs([]))

    stream = llm_model.chat(
        chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0, timeout=30)
    )
    with pytest.raises(APITimeoutError):
        await asyncio.wait_for(_consume(stream), 5)
    await stream.aclose()
    await llm_model.aclose()


async def test_ws_pongs_do_not_reset_idle_timeout() -> None:
    llm_model = _make_llm(stream_idle_timeout=0.1)
    ws = _PongingWs()
    _use_connection(llm_model, ws)

    stream = llm_model.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0))
    with pytest.raises(APITimeoutError):
        await asyncio.wait_for(_consume(stream), 5)
    await stream.aclose()

    assert ws.pongs > 0
    assert ws not in _transport(llm_model)._pool._available
    await llm_model.aclose()


async def test_ws_text_frame_resets_idle_timeout() -> None:
    llm_model = _make_llm(stream_idle_timeout=0.2)
    ws = _ScriptedWs([_CONNECTED, *_TEXT_FRAMES])
    scripted_receive = ws.receive

    async def _slow_receive(timeout: float | None = None) -> _FakeWsMsg:
        # each frame lands inside the window, though together they outlast it
        await asyncio.sleep(0.08)
        return await scripted_receive(timeout)

    ws.receive = _slow_receive  # type: ignore[method-assign]
    _use_connection(llm_model, ws)

    assert await _drain(llm_model) == "hi"
    await llm_model.aclose()


async def test_ws_dial_uses_request_connect_timeout() -> None:
    llm_model = _make_llm()
    ws = _ScriptedWs([_CONNECTED, *_TEXT_FRAMES])
    timeouts: list[float] = []

    async def _connect(timeout: float) -> _ScriptedWs:
        timeouts.append(timeout)
        return ws

    _transport(llm_model)._pool._connect_cb = _connect  # type: ignore[assignment]

    stream = llm_model.chat(
        chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0, timeout=1.5)
    )
    async for _ in stream:
        pass
    await stream.aclose()

    assert timeouts == [1.5]
    await llm_model.aclose()


async def test_ws_dial_sends_constructor_extra_headers() -> None:
    llm_model = _make_llm(extra_kwargs={"extra_headers": {"X-Deployment": "blue"}})
    session = _dial_session(llm_model)

    await _transport(llm_model)._create_ws(1.0)

    assert session.dials[0].headers["X-Deployment"] == "blue"
    assert session.dials[0].headers["Authorization"].startswith("Bearer ")
    await llm_model.aclose()


async def test_ws_per_call_extra_headers_warns(caplog: pytest.LogCaptureFixture) -> None:
    llm_model = _make_llm()

    with caplog.at_level(logging.WARNING):
        stream = llm_model.chat(
            chat_ctx=_ctx(), extra_kwargs={"extra_headers": {"X-Deployment": "blue"}}
        )
    await stream.aclose()

    records = [r for r in caplog.records if "extra_headers cannot change" in r.getMessage()]
    assert len(records) == 1
    await llm_model.aclose()


async def test_gateway_upstream_connect_failed_is_retryable_502() -> None:
    llm_model = _make_llm()
    upstream_failed = {
        "type": "gateway.error",
        "code": "upstream_connect_failed",
        "message": "all 1 responses ws candidates failed to dial",
    }
    _use_connection(llm_model, _ScriptedWs([upstream_failed]))

    with pytest.raises(APIStatusError) as exc_info:
        await _drain(llm_model)

    assert "upstream_connect_failed" in exc_info.value.message
    assert exc_info.value.status_code == 502
    assert exc_info.value.retryable is True
    await llm_model.aclose()


async def test_handshake_401_is_api_status_error_not_retryable() -> None:
    llm_model = _make_llm()
    _transport(llm_model)._session = _RefusingSession(401)  # type: ignore[assignment]

    with pytest.raises(APIStatusError) as exc_info:
        await _transport(llm_model)._create_ws(1.0)

    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False
    await llm_model.aclose()


async def test_handshake_429_logs_quota_headers(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    llm_model = _make_llm()
    _transport(llm_model)._session = _RefusingSession(  # type: ignore[assignment]
        429, {"X-LiveKit-Inference-RPM-Limit": "10"}
    )

    with pytest.raises(APIStatusError) as exc_info:
        await _transport(llm_model)._create_ws(1.0)

    records = [
        r
        for r in caplog.records
        if r.getMessage() == "LLM request rate limited by inference gateway"
    ]
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True
    assert len(records) == 1
    assert records[0].rpm_limit == "10"  # type: ignore[attr-defined]
    assert records[0].model == "openai/gpt-4.1"  # type: ignore[attr-defined]
    await llm_model.aclose()


async def test_ws_receive_timeout_raises_api_timeout() -> None:
    llm_model = _make_llm()
    ws = _SilentWs([])
    _use_connection(llm_model, ws)

    stream = llm_model.chat(
        chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0, timeout=0.05)
    )
    with pytest.raises(APITimeoutError):
        async for _ in stream:
            pass
    await stream.aclose()

    assert ws not in _transport(llm_model)._pool._available
    assert ws not in _transport(llm_model)._pool._connections
    await llm_model.aclose()


async def test_gateway_error_socket_leaves_pool() -> None:
    llm_model = _make_llm()
    transport = _transport(llm_model)
    ws = _ScriptedWs([_CONNECTED, _INVALID_EVENT])
    _use_connection(llm_model, ws)

    # the stream stops reading at the error and drops the generator; close it the same way
    events = transport.generate_response({"type": "response.create"})
    assert (await events.__anext__())["type"] == "gateway.connected"
    frame = await events.__anext__()
    parsed = inference_responses._GatewayResponsesLLMStream._parse_ws_event(object(), frame)  # type: ignore[arg-type]
    await events.aclose()

    assert parsed is not None and parsed.type == "error"
    assert ws not in transport._pool._available
    assert ws not in transport._pool._connections
    await llm_model.aclose()


async def test_gateway_connected_is_ignored() -> None:
    llm_model = _make_llm()
    _use_connection(llm_model, _ScriptedWs([_CONNECTED, *_TEXT_FRAMES]))

    text = await _drain(llm_model)

    assert text == "hi"
    await llm_model.aclose()


async def test_gateway_session_expiring_is_ignored() -> None:
    llm_model = _make_llm()
    expiring = {"type": "gateway.session_expiring", "seconds_until_close": 300}
    _use_connection(llm_model, _ScriptedWs([_CONNECTED, expiring, *_TEXT_FRAMES]))

    text = await _drain(llm_model)

    assert text == "hi"
    await llm_model.aclose()


async def test_ws_pool_session_is_55_minutes() -> None:
    llm_model = _make_llm()

    assert _transport(llm_model)._pool._max_session_duration == 3300
    await llm_model.aclose()


async def test_ws_payload_has_no_extra_headers() -> None:
    llm_model = _make_llm(provider="azure", inference_class="priority")
    ws = _ScriptedWs([_CONNECTED, *_TEXT_FRAMES])
    _use_connection(llm_model, ws)

    await _drain(llm_model)

    payload = json.loads(ws.sent[0])
    assert payload["type"] == "response.create"
    assert "extra_headers" not in payload
    await llm_model.aclose()


async def test_websocket_mode_builds_no_rest_client() -> None:
    llm_model = _make_llm()

    assert llm_model._opts.use_websocket is True
    assert llm_model._client is None
    assert _transport(llm_model)._session is None
    await llm_model.aclose()


async def test_engine_default_session_duration_unchanged() -> None:
    transport = ResponsesWebsocket(api_key="k", timeout=1.0, model="gpt-4.1")

    assert transport._pool._max_session_duration == 3600
    await transport.aclose()
